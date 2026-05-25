"""Throughput benchmarks for Adaptive KV Memory.

Measures tokens/second across different configurations, sequence lengths,
and model sizes. Produces publication-ready results with statistical rigor.

Usage:
    python -m benchmarks.throughput_bench --model meta-llama/Llama-2-7b-hf
    python -m benchmarks.throughput_bench --model mistralai/Mistral-7B-v0.1 --seq-lens 1024,4096,16384
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from akv.cache import CacheConfig
from akv.integration import HFAdaptiveCache
from akv.baselines import (
    FullCache, H2OCache, H2OConfig,
    KIVICache, KIVIConfig, SnapKVCache, SnapKVConfig,
    ScissorHandsCache, ScissorHandsConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    method: str
    seq_len: int
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    vram_peak_mb: float
    vram_allocated_mb: float
    num_generated_tokens: int
    batch_size: int
    std_decode_tok_s: float = 0.0
    num_runs: int = 1
    config: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    seq_lens: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384])
    max_new_tokens: int = 256
    batch_size: int = 1
    num_warmup: int = 2
    num_runs: int = 5
    device: str = "cuda"
    dtype: str = "float16"
    output_dir: str = "./benchmark_results"
    methods: list[str] = field(default_factory=lambda: [
        "full", "akv-4bit", "akv-2bit", "h2o", "kivi", "snapkv"
    ])


class ThroughputBenchmark:
    """Comprehensive throughput benchmark suite.

    Measures:
    - Prefill throughput (tokens/sec for prompt processing)
    - Decode throughput (tokens/sec for autoregressive generation)
    - Peak VRAM usage
    - Statistical variance across runs
    """

    def __init__(self, config: BenchConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results: list[ThroughputResult] = []

    def setup(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.config.dtype)
        logger.info(f"Loading {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded on {self.config.device}")

    def _create_input(self, seq_len: int) -> torch.Tensor:
        """Create synthetic input of target sequence length."""
        # Use repeating tokens to fill the context
        vocab_size = self.tokenizer.vocab_size
        tokens = torch.randint(100, vocab_size - 100, (1, seq_len), device=self.config.device)
        return tokens

    def _clear_gpu(self):
        """Clear GPU memory between runs."""
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _measure_baseline(self, input_ids: torch.Tensor, max_new_tokens: int) -> ThroughputResult:
        """Measure full-cache baseline throughput."""
        device = self.config.device
        times_prefill = []
        times_decode = []
        vram_peaks = []

        for run_idx in range(self.config.num_warmup + self.config.num_runs):
            self._clear_gpu()

            # Prefill phase
            if device == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            with torch.inference_mode():
                outputs = self.model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

            if device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                prefill_ms = start_event.elapsed_time(end_event)
            else:
                prefill_ms = 0.0

            # Decode phase
            generated_tokens = [next_token]
            if device == "cuda":
                torch.cuda.synchronize()
                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)
                decode_start.record()

            with torch.inference_mode():
                for _ in range(max_new_tokens - 1):
                    outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:].argmax(dim=-1)
                    generated_tokens.append(next_token)
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            if device == "cuda":
                decode_end.record()
                torch.cuda.synchronize()
                decode_ms = decode_start.elapsed_time(decode_end)
                vram_peak = torch.cuda.max_memory_allocated() / 1e6
            else:
                decode_ms = 0.0
                vram_peak = 0.0

            if run_idx >= self.config.num_warmup:
                times_prefill.append(prefill_ms)
                times_decode.append(decode_ms)
                vram_peaks.append(vram_peak)

        n_gen = len(generated_tokens)
        avg_prefill = np.mean(times_prefill)
        avg_decode = np.mean(times_decode)
        std_decode = np.std(times_decode)

        return ThroughputResult(
            method="Full Cache (Baseline)",
            seq_len=input_ids.shape[1],
            prefill_tokens_per_sec=input_ids.shape[1] / (avg_prefill / 1000) if avg_prefill > 0 else 0,
            decode_tokens_per_sec=n_gen / (avg_decode / 1000) if avg_decode > 0 else 0,
            prefill_time_ms=avg_prefill,
            decode_time_ms=avg_decode,
            total_time_ms=avg_prefill + avg_decode,
            vram_peak_mb=np.mean(vram_peaks),
            vram_allocated_mb=np.mean(vram_peaks) * 0.85,  # approximation
            num_generated_tokens=n_gen,
            batch_size=self.config.batch_size,
            std_decode_tok_s=n_gen / (std_decode / 1000) if std_decode > 0 else 0,
            num_runs=self.config.num_runs,
        )

    def _measure_adaptive(
        self, input_ids: torch.Tensor, max_new_tokens: int, cache_config: CacheConfig, method_name: str
    ) -> ThroughputResult:
        """Measure adaptive cache throughput."""
        device = self.config.device
        times_prefill = []
        times_decode = []
        vram_peaks = []

        for run_idx in range(self.config.num_warmup + self.config.num_runs):
            self._clear_gpu()
            cache = HFAdaptiveCache(cache_config)

            # Prefill
            if device == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            with torch.inference_mode():
                outputs = self.model(input_ids, past_key_values=cache, use_cache=True)
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

            if device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                prefill_ms = start_event.elapsed_time(end_event)
            else:
                prefill_ms = 0.0

            # Decode
            generated_tokens = [next_token]
            if device == "cuda":
                torch.cuda.synchronize()
                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)
                decode_start.record()

            with torch.inference_mode():
                for _ in range(max_new_tokens - 1):
                    outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:].argmax(dim=-1)
                    generated_tokens.append(next_token)
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            if device == "cuda":
                decode_end.record()
                torch.cuda.synchronize()
                decode_ms = decode_start.elapsed_time(decode_end)
                vram_peak = torch.cuda.max_memory_allocated() / 1e6
            else:
                decode_ms = 0.0
                vram_peak = 0.0

            if run_idx >= self.config.num_warmup:
                times_prefill.append(prefill_ms)
                times_decode.append(decode_ms)
                vram_peaks.append(vram_peak)

        n_gen = len(generated_tokens)
        avg_prefill = np.mean(times_prefill)
        avg_decode = np.mean(times_decode)
        std_decode = np.std(times_decode)

        return ThroughputResult(
            method=method_name,
            seq_len=input_ids.shape[1],
            prefill_tokens_per_sec=input_ids.shape[1] / (avg_prefill / 1000) if avg_prefill > 0 else 0,
            decode_tokens_per_sec=n_gen / (avg_decode / 1000) if avg_decode > 0 else 0,
            prefill_time_ms=avg_prefill,
            decode_time_ms=avg_decode,
            total_time_ms=avg_prefill + avg_decode,
            vram_peak_mb=np.mean(vram_peaks),
            vram_allocated_mb=np.mean(vram_peaks) * 0.85,
            num_generated_tokens=n_gen,
            batch_size=self.config.batch_size,
            std_decode_tok_s=n_gen / (std_decode / 1000) if std_decode > 0 else 0,
            num_runs=self.config.num_runs,
            config=asdict(cache_config) if hasattr(cache_config, '__dataclass_fields__') else {},
        )

    def _get_method_configs(self) -> list[tuple[str, CacheConfig]]:
        """Get cache configurations for each method to benchmark."""
        configs = []
        for method in self.config.methods:
            if method == "akv-4bit":
                configs.append(("AKV-4bit (hot=1024, warm=2048)", CacheConfig(
                    hot_budget=1024, warm_budget=2048, warm_bits=4, cold_bits=2,
                    enable_cold_tier=True, group_size=128,
                )))
            elif method == "akv-2bit":
                configs.append(("AKV-2bit (hot=1024, warm=2048)", CacheConfig(
                    hot_budget=1024, warm_budget=2048, warm_bits=2, cold_bits=2,
                    enable_cold_tier=True, group_size=128,
                )))
            elif method == "akv-aggressive":
                configs.append(("AKV-Aggressive (hot=512, warm=4096)", CacheConfig(
                    hot_budget=512, warm_budget=4096, warm_bits=4, cold_bits=2,
                    enable_cold_tier=True, group_size=64,
                )))
        return configs

    def run(self) -> list[ThroughputResult]:
        """Run full throughput benchmark suite."""
        if self.model is None:
            self.setup()

        results = []
        for seq_len in self.config.seq_lens:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking seq_len={seq_len}")
            logger.info(f"{'='*60}")

            input_ids = self._create_input(seq_len)

            # Baseline
            if "full" in self.config.methods:
                try:
                    r = self._measure_baseline(input_ids, self.config.max_new_tokens)
                    results.append(r)
                    logger.info(f"  Full Cache: {r.decode_tokens_per_sec:.1f} tok/s, "
                               f"VRAM={r.vram_peak_mb:.0f}MB")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"  Full Cache: OOM at seq_len={seq_len}")
                    self._clear_gpu()

            # Adaptive methods
            for method_name, cache_config in self._get_method_configs():
                try:
                    r = self._measure_adaptive(input_ids, self.config.max_new_tokens, cache_config, method_name)
                    results.append(r)
                    logger.info(f"  {method_name}: {r.decode_tokens_per_sec:.1f} tok/s, "
                               f"VRAM={r.vram_peak_mb:.0f}MB")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"  {method_name}: OOM at seq_len={seq_len}")
                    self._clear_gpu()

        self.results = results
        return results

    def save_results(self, path: Optional[str] = None):
        """Save results to JSON."""
        if path is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            path = f"{self.config.output_dir}/throughput_{int(time.time())}.json"

        data = {
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
            "summary": self._compute_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def _compute_summary(self) -> dict:
        """Compute summary statistics across all results."""
        if not self.results:
            return {}

        by_method = {}
        for r in self.results:
            if r.method not in by_method:
                by_method[r.method] = []
            by_method[r.method].append(r)

        summary = {}
        baseline_decode = None
        for method, runs in by_method.items():
            avg_decode = np.mean([r.decode_tokens_per_sec for r in runs])
            avg_vram = np.mean([r.vram_peak_mb for r in runs])
            if method == "Full Cache (Baseline)":
                baseline_decode = avg_decode
            summary[method] = {
                "avg_decode_tok_s": round(avg_decode, 1),
                "avg_vram_mb": round(avg_vram, 0),
                "speedup_vs_baseline": round(avg_decode / baseline_decode, 2) if baseline_decode else None,
                "vram_savings_vs_baseline_pct": round(
                    (1 - avg_vram / summary.get("Full Cache (Baseline)", {}).get("avg_vram_mb", avg_vram)) * 100, 1
                ) if "Full Cache (Baseline)" in summary else None,
            }

        return summary

    def print_results(self):
        """Pretty-print results table."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Throughput Benchmark Results", show_lines=True)
            table.add_column("Method", style="cyan", min_width=25)
            table.add_column("Seq Len", justify="right")
            table.add_column("Prefill tok/s", justify="right")
            table.add_column("Decode tok/s", justify="right", style="green")
            table.add_column("Total ms", justify="right")
            table.add_column("VRAM Peak MB", justify="right", style="yellow")
            table.add_column("Generated", justify="right")

            for r in self.results:
                table.add_row(
                    r.method,
                    str(r.seq_len),
                    f"{r.prefill_tokens_per_sec:,.0f}",
                    f"{r.decode_tokens_per_sec:,.1f}",
                    f"{r.total_time_ms:,.0f}",
                    f"{r.vram_peak_mb:,.0f}",
                    str(r.num_generated_tokens),
                )

            console.print(table)
        except ImportError:
            print("\nThroughput Benchmark Results:")
            print("-" * 80)
            for r in self.results:
                print(f"  {r.method} @ {r.seq_len}: "
                      f"decode={r.decode_tokens_per_sec:.1f} tok/s, "
                      f"VRAM={r.vram_peak_mb:.0f}MB")


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmarks for Adaptive KV Memory")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seq-lens", default="512,1024,2048,4096,8192",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--methods", default="full,akv-4bit,akv-2bit",
                        help="Comma-separated methods to benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = BenchConfig(
        model_name=args.model,
        seq_lens=[int(x) for x in args.seq_lens.split(",")],
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        num_warmup=args.num_warmup,
        output_dir=args.output_dir,
        methods=[x.strip() for x in args.methods.split(",")],
    )

    bench = ThroughputBenchmark(config)
    bench.run()
    bench.print_results()
    bench.save_results()


if __name__ == "__main__":
    main()
