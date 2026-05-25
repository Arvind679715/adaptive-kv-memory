"""Latency benchmarks for Adaptive KV Memory.

Measures per-token latency (time-to-first-token, inter-token latency),
P50/P95/P99 percentiles, and latency distribution under various conditions.

Key metrics:
- TTFT (Time to First Token): latency from request start to first generated token
- ITL (Inter-Token Latency): time between consecutive generated tokens
- E2E (End-to-End): total time from request to last token

Usage:
    python -m benchmarks.latency_bench --model meta-llama/Llama-2-7b-hf
    python -m benchmarks.latency_bench --model mistralai/Mistral-7B-v0.1 --profile
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

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    method: str
    seq_len: int
    # Time to first token (prefill latency)
    ttft_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    # Inter-token latency (decode latency per token)
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_std_ms: float
    # End-to-end
    e2e_ms: float
    # Per-step breakdown (for latency profiles)
    per_token_ms: list[float] = field(default_factory=list)
    # Tier migration overhead
    migration_events: int = 0
    avg_migration_ms: float = 0.0
    # Memory
    vram_peak_mb: float = 0.0
    num_generated: int = 0
    num_runs: int = 1

    def to_dict(self):
        d = asdict(self)
        # Truncate per_token_ms for storage
        if len(d["per_token_ms"]) > 100:
            d["per_token_ms_sample"] = d["per_token_ms"][:50] + d["per_token_ms"][-50:]
            d["per_token_ms_full_len"] = len(d["per_token_ms"])
            del d["per_token_ms"]
        return d


@dataclass
class LatencyBenchConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    seq_lens: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    max_new_tokens: int = 128
    num_warmup: int = 3
    num_runs: int = 10
    device: str = "cuda"
    dtype: str = "float16"
    output_dir: str = "./benchmark_results"
    profile_per_token: bool = True  # Record per-token latencies
    methods: list[str] = field(default_factory=lambda: [
        "full", "akv-4bit", "akv-2bit"
    ])


class LatencyBenchmark:
    """Fine-grained latency benchmark suite.

    Goes beyond simple throughput to measure:
    - TTFT: critical for interactive applications
    - ITL distribution: reveals jitter from tier migrations
    - Tail latencies: P95/P99 showing worst-case behavior
    - Per-token latency traces: shows when migrations cause spikes
    """

    def __init__(self, config: LatencyBenchConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results: list[LatencyResult] = []

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
        logger.info("Model loaded.")

    def _clear_gpu(self):
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _create_input(self, seq_len: int) -> torch.Tensor:
        vocab_size = self.tokenizer.vocab_size
        return torch.randint(100, vocab_size - 100, (1, seq_len), device=self.config.device)

    def _measure_latency_baseline(self, input_ids: torch.Tensor) -> LatencyResult:
        """Measure per-token latency for full cache baseline."""
        device = self.config.device
        max_new_tokens = self.config.max_new_tokens
        all_ttft = []
        all_itls = []
        all_e2e = []
        all_per_token = []

        for run_idx in range(self.config.num_warmup + self.config.num_runs):
            self._clear_gpu()
            per_token_times = []

            # TTFT: time to first token (prefill)
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

            if device == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter()
            ttft = (t_first - t_start) * 1000  # ms

            # Decode: measure each token's latency individually
            itls = []
            for step in range(max_new_tokens - 1):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.inference_mode():
                    outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:].argmax(dim=-1)

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                itl_ms = (t1 - t0) * 1000
                itls.append(itl_ms)
                per_token_times.append(itl_ms)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            e2e = (t_end - t_start) * 1000

            if run_idx >= self.config.num_warmup:
                all_ttft.append(ttft)
                all_itls.extend(itls)
                all_e2e.append(e2e)
                all_per_token = per_token_times  # Keep last run's trace

        vram_peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

        itl_array = np.array(all_itls)
        return LatencyResult(
            method="Full Cache (Baseline)",
            seq_len=input_ids.shape[1],
            ttft_ms=np.mean(all_ttft),
            ttft_p50_ms=np.percentile(all_ttft, 50),
            ttft_p95_ms=np.percentile(all_ttft, 95),
            ttft_p99_ms=np.percentile(all_ttft, 99),
            itl_mean_ms=np.mean(itl_array),
            itl_p50_ms=np.percentile(itl_array, 50),
            itl_p95_ms=np.percentile(itl_array, 95),
            itl_p99_ms=np.percentile(itl_array, 99),
            itl_std_ms=np.std(itl_array),
            e2e_ms=np.mean(all_e2e),
            per_token_ms=all_per_token if self.config.profile_per_token else [],
            vram_peak_mb=vram_peak,
            num_generated=len(all_per_token),
            num_runs=self.config.num_runs,
        )

    def _measure_latency_adaptive(
        self, input_ids: torch.Tensor, cache_config: CacheConfig, method_name: str
    ) -> LatencyResult:
        """Measure per-token latency for adaptive cache."""
        device = self.config.device
        max_new_tokens = self.config.max_new_tokens
        all_ttft = []
        all_itls = []
        all_e2e = []
        all_per_token = []
        migration_count = 0

        for run_idx in range(self.config.num_warmup + self.config.num_runs):
            self._clear_gpu()
            cache = HFAdaptiveCache(cache_config)
            per_token_times = []

            # TTFT
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model(input_ids, past_key_values=cache, use_cache=True)
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

            if device == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter()
            ttft = (t_first - t_start) * 1000

            # Decode with per-token measurement
            itls = []
            for step in range(max_new_tokens - 1):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.inference_mode():
                    outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_token = outputs.logits[:, -1:].argmax(dim=-1)

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                itl_ms = (t1 - t0) * 1000
                itls.append(itl_ms)
                per_token_times.append(itl_ms)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            e2e = (t_end - t_start) * 1000

            if run_idx >= self.config.num_warmup:
                all_ttft.append(ttft)
                all_itls.extend(itls)
                all_e2e.append(e2e)
                all_per_token = per_token_times

                # Track migration events from cache stats
                tier_info = cache.tier_summary()
                migration_count += tier_info.get("migrations", 0)

        vram_peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

        itl_array = np.array(all_itls)
        return LatencyResult(
            method=method_name,
            seq_len=input_ids.shape[1],
            ttft_ms=np.mean(all_ttft),
            ttft_p50_ms=np.percentile(all_ttft, 50),
            ttft_p95_ms=np.percentile(all_ttft, 95),
            ttft_p99_ms=np.percentile(all_ttft, 99),
            itl_mean_ms=np.mean(itl_array),
            itl_p50_ms=np.percentile(itl_array, 50),
            itl_p95_ms=np.percentile(itl_array, 95),
            itl_p99_ms=np.percentile(itl_array, 99),
            itl_std_ms=np.std(itl_array),
            e2e_ms=np.mean(all_e2e),
            per_token_ms=all_per_token if self.config.profile_per_token else [],
            migration_events=migration_count,
            vram_peak_mb=vram_peak,
            num_generated=len(all_per_token),
            num_runs=self.config.num_runs,
        )

    def run(self) -> list[LatencyResult]:
        """Run full latency benchmark suite."""
        if self.model is None:
            self.setup()

        results = []
        for seq_len in self.config.seq_lens:
            logger.info(f"\n{'='*60}")
            logger.info(f"Latency benchmark: seq_len={seq_len}")
            logger.info(f"{'='*60}")

            input_ids = self._create_input(seq_len)

            # Baseline
            if "full" in self.config.methods:
                try:
                    r = self._measure_latency_baseline(input_ids)
                    results.append(r)
                    logger.info(f"  Full Cache: TTFT={r.ttft_ms:.1f}ms, "
                               f"ITL_p50={r.itl_p50_ms:.2f}ms, "
                               f"ITL_p99={r.itl_p99_ms:.2f}ms")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"  Full Cache: OOM at seq_len={seq_len}")
                    self._clear_gpu()

            # Adaptive methods
            method_configs = {
                "akv-4bit": ("AKV-4bit", CacheConfig(
                    hot_budget=1024, warm_budget=2048, warm_bits=4, cold_bits=2,
                    enable_cold_tier=True, group_size=128,
                )),
                "akv-2bit": ("AKV-2bit", CacheConfig(
                    hot_budget=1024, warm_budget=2048, warm_bits=2, cold_bits=2,
                    enable_cold_tier=True, group_size=128,
                )),
            }

            for method_key in self.config.methods:
                if method_key in method_configs:
                    name, cfg = method_configs[method_key]
                    try:
                        r = self._measure_latency_adaptive(input_ids, cfg, name)
                        results.append(r)
                        logger.info(f"  {name}: TTFT={r.ttft_ms:.1f}ms, "
                                   f"ITL_p50={r.itl_p50_ms:.2f}ms, "
                                   f"ITL_p99={r.itl_p99_ms:.2f}ms, "
                                   f"migrations={r.migration_events}")
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"  {name}: OOM at seq_len={seq_len}")
                        self._clear_gpu()

        self.results = results
        return results

    def save_results(self, path: Optional[str] = None):
        """Save results to JSON."""
        if path is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            path = f"{self.config.output_dir}/latency_{int(time.time())}.json"

        data = {
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
            "summary": self._compute_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def _compute_summary(self) -> dict:
        """Summary statistics."""
        summary = {}
        for r in self.results:
            key = f"{r.method}@{r.seq_len}"
            summary[key] = {
                "ttft_ms": round(r.ttft_ms, 2),
                "itl_p50_ms": round(r.itl_p50_ms, 3),
                "itl_p95_ms": round(r.itl_p95_ms, 3),
                "itl_p99_ms": round(r.itl_p99_ms, 3),
                "e2e_ms": round(r.e2e_ms, 1),
                "vram_mb": round(r.vram_peak_mb, 0),
            }
        return summary

    def print_results(self):
        """Pretty-print latency results."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Latency Benchmark Results (ms)", show_lines=True)
            table.add_column("Method", style="cyan", min_width=20)
            table.add_column("Seq Len", justify="right")
            table.add_column("TTFT", justify="right", style="yellow")
            table.add_column("ITL p50", justify="right", style="green")
            table.add_column("ITL p95", justify="right")
            table.add_column("ITL p99", justify="right", style="red")
            table.add_column("ITL std", justify="right")
            table.add_column("E2E", justify="right")
            table.add_column("VRAM MB", justify="right")

            for r in self.results:
                table.add_row(
                    r.method, str(r.seq_len),
                    f"{r.ttft_ms:.1f}",
                    f"{r.itl_p50_ms:.2f}",
                    f"{r.itl_p95_ms:.2f}",
                    f"{r.itl_p99_ms:.2f}",
                    f"{r.itl_std_ms:.2f}",
                    f"{r.e2e_ms:.0f}",
                    f"{r.vram_peak_mb:.0f}",
                )

            console.print(table)
        except ImportError:
            print("\nLatency Benchmark Results:")
            print("-" * 90)
            for r in self.results:
                print(f"  {r.method} @ {r.seq_len}: "
                      f"TTFT={r.ttft_ms:.1f}ms, "
                      f"ITL[p50={r.itl_p50_ms:.2f}, p95={r.itl_p95_ms:.2f}, p99={r.itl_p99_ms:.2f}]ms, "
                      f"VRAM={r.vram_peak_mb:.0f}MB")

    def plot_latency_trace(self, output_path: Optional[str] = None):
        """Plot per-token latency trace showing migration spikes."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(self.config.seq_lens), 1,
                                     figsize=(12, 4 * len(self.config.seq_lens)))
            if len(self.config.seq_lens) == 1:
                axes = [axes]

            for ax, seq_len in zip(axes, self.config.seq_lens):
                seq_results = [r for r in self.results if r.seq_len == seq_len]
                for r in seq_results:
                    if r.per_token_ms:
                        ax.plot(r.per_token_ms, label=r.method, alpha=0.8)
                ax.set_xlabel("Token Position")
                ax.set_ylabel("Latency (ms)")
                ax.set_title(f"Per-Token Latency @ seq_len={seq_len}")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"Latency trace plot saved to {output_path}")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib not available for plotting")


def main():
    parser = argparse.ArgumentParser(description="Latency benchmarks for Adaptive KV Memory")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seq-lens", default="512,1024,2048,4096,8192")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--profile", action="store_true", help="Enable per-token profiling")
    parser.add_argument("--plot", action="store_true", help="Generate latency trace plots")
    parser.add_argument("--methods", default="full,akv-4bit,akv-2bit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = LatencyBenchConfig(
        model_name=args.model,
        seq_lens=[int(x) for x in args.seq_lens.split(",")],
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        profile_per_token=args.profile,
        methods=[x.strip() for x in args.methods.split(",")],
    )

    bench = LatencyBenchmark(config)
    bench.run()
    bench.print_results()
    bench.save_results()

    if args.plot:
        bench.plot_latency_trace(f"{args.output_dir}/latency_trace.png")


if __name__ == "__main__":
    main()
