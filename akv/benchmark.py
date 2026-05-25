"""Benchmark suite for Adaptive KV Memory.

Measures VRAM usage, throughput, perplexity preservation, and
context length extension compared to standard DynamicCache.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    name: str
    baseline_tok_s: float = 0.0
    adaptive_tok_s: float = 0.0
    speedup: float = 0.0
    baseline_vram_mb: float = 0.0
    adaptive_vram_mb: float = 0.0
    vram_savings_mb: float = 0.0
    vram_savings_pct: float = 0.0
    baseline_ppl: float = 0.0
    adaptive_ppl: float = 0.0
    ppl_ratio: float = 0.0
    context_len: int = 0
    max_new_tokens: int = 0
    tier_summary: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Benchmark:
    """Runs comprehensive benchmarks comparing adaptive cache vs baseline."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_throughput(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        num_runs: int = 3,
        cache_config=None,
    ) -> list[BenchmarkResult]:
        """Benchmark throughput: tokens/second for baseline vs adaptive."""
        from akv.integration import HFAdaptiveCache

        results = []

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            name = prompt[:40].replace('\n', ' ')

            # --- Baseline (standard DynamicCache) ---
            base_times = []
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                base_times.append(t1 - t0)
            base_tokens = out.shape[1] - input_ids.shape[1]
            base_avg = np.mean(base_times)
            base_vram = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0

            # --- Adaptive cache ---
            adap_times = []
            final_cache = None
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                cache = HFAdaptiveCache(cache_config)
                t0 = time.perf_counter()
                generated = self._generate_with_cache(input_ids, cache, max_new_tokens)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                adap_times.append(t1 - t0)
                final_cache = cache
            adap_tokens = len(generated)
            adap_avg = np.mean(adap_times)
            adap_vram = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0

            r = BenchmarkResult(
                name=name,
                baseline_tok_s=round(base_tokens / max(base_avg, 1e-6), 1),
                adaptive_tok_s=round(adap_tokens / max(adap_avg, 1e-6), 1),
                speedup=round(base_avg / max(adap_avg, 1e-6), 2),
                baseline_vram_mb=round(base_vram, 1),
                adaptive_vram_mb=round(adap_vram, 1),
                vram_savings_mb=round(base_vram - adap_vram, 1),
                vram_savings_pct=round((base_vram - adap_vram) / max(base_vram, 1) * 100, 1),
                context_len=input_ids.shape[1],
                max_new_tokens=max_new_tokens,
                tier_summary=final_cache.tier_summary() if final_cache else {},
            )
            results.append(r)

        return results

    def run_memory_scaling(
        self,
        base_prompt: str,
        context_lengths: list[int],
        cache_config=None,
    ) -> list[dict]:
        """Measure VRAM usage at different context lengths."""
        from akv.integration import HFAdaptiveCache

        results = []
        for ctx_len in context_lengths:
            # Repeat prompt to fill context
            tokens = self.tokenizer.encode(base_prompt)
            repeated = (tokens * ((ctx_len // len(tokens)) + 1))[:ctx_len]
            input_ids = torch.tensor([repeated], device=self.device)

            # Baseline
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                try:
                    self.model.generate(input_ids, max_new_tokens=1, do_sample=False)
                    base_vram = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0
                    base_oom = False
                except torch.cuda.OutOfMemoryError:
                    base_vram = -1
                    base_oom = True
                    torch.cuda.empty_cache()

            # Adaptive
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            cache = HFAdaptiveCache(cache_config)
            try:
                self._generate_with_cache(input_ids, cache, max_new_tokens=1)
                adap_vram = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0
                adap_oom = False
            except torch.cuda.OutOfMemoryError:
                adap_vram = -1
                adap_oom = True
                torch.cuda.empty_cache()

            results.append({
                "context_length": ctx_len,
                "baseline_vram_mb": round(base_vram, 1),
                "adaptive_vram_mb": round(adap_vram, 1),
                "baseline_oom": base_oom,
                "adaptive_oom": adap_oom,
                "savings_mb": round(base_vram - adap_vram, 1) if not (base_oom or adap_oom) else None,
                "tier_summary": cache.tier_summary(),
            })

        return results

    def run_perplexity(
        self,
        texts: list[str],
        cache_config=None,
    ) -> list[dict]:
        """Compare output perplexity between baseline and adaptive generation."""
        from akv.integration import HFAdaptiveCache

        results = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

            # Generate with both methods
            with torch.inference_mode():
                base_out = self.model.generate(input_ids, max_new_tokens=64, do_sample=False)
            base_text = self.tokenizer.decode(base_out[0], skip_special_tokens=True)

            cache = HFAdaptiveCache(cache_config)
            adap_tokens = self._generate_with_cache(input_ids, cache, max_new_tokens=64)
            adap_text = self.tokenizer.decode(adap_tokens, skip_special_tokens=True)

            # Score both
            base_ppl = self._compute_ppl(base_text)
            adap_ppl = self._compute_ppl(adap_text)

            results.append({
                "prompt": text[:60],
                "baseline_ppl": round(base_ppl, 2),
                "adaptive_ppl": round(adap_ppl, 2),
                "ppl_ratio": round(adap_ppl / max(base_ppl, 1e-6), 3),
                "baseline_text": base_text[:200],
                "adaptive_text": adap_text[:200],
            })

        return results

    def _generate_with_cache(self, input_ids, cache, max_new_tokens):
        """Manual generation loop using our adaptive cache."""
        generated = []
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())

            for _ in range(max_new_tokens - 1):
                outputs = self.model(input_ids=next_token, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token.item())
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return generated

    def _compute_ppl(self, text: str) -> float:
        """Compute perplexity of a text string."""
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if tokens.shape[1] < 2:
            return 0.0
        with torch.inference_mode():
            outputs = self.model(tokens)
            logits = outputs.logits
        log_probs = logits[:, :-1, :].log_softmax(dim=-1)
        targets = tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, targets).squeeze(-1)
        return torch.exp(-token_log_probs.mean()).item()

    @staticmethod
    def save_results(results: list, path: str):
        """Save benchmark results to JSON."""
        serializable = []
        for r in results:
            if isinstance(r, BenchmarkResult):
                serializable.append(r.to_dict())
            else:
                serializable.append(r)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    @staticmethod
    def print_results(results: list):
        """Pretty-print benchmark results."""
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()

            table = Table(title="Adaptive KV Memory Benchmark Results")
            table.add_column("Test", style="cyan")
            table.add_column("Baseline tok/s", justify="right")
            table.add_column("Adaptive tok/s", justify="right")
            table.add_column("Speedup", justify="right")
            table.add_column("VRAM Saved", justify="right")
            table.add_column("PPL Ratio", justify="right")

            for r in results:
                if isinstance(r, BenchmarkResult):
                    table.add_row(
                        r.name,
                        f"{r.baseline_tok_s:.1f}",
                        f"{r.adaptive_tok_s:.1f}",
                        f"{r.speedup:.2f}x",
                        f"{r.vram_savings_mb:+.0f} MB ({r.vram_savings_pct:.1f}%)",
                        f"{r.ppl_ratio:.3f}x" if r.ppl_ratio > 0 else "N/A",
                    )

            console.print(table)
        except ImportError:
            for r in results:
                if isinstance(r, BenchmarkResult):
                    print(f"\n[{r.name}]")
                    print(f"  Baseline: {r.baseline_tok_s:.1f} tok/s | VRAM: {r.baseline_vram_mb:.0f} MB")
                    print(f"  Adaptive: {r.adaptive_tok_s:.1f} tok/s | VRAM: {r.adaptive_vram_mb:.0f} MB")
                    print(f"  Speedup: {r.speedup:.2f}x | VRAM saved: {r.vram_savings_mb:+.0f} MB")
                elif isinstance(r, dict):
                    print(f"\n  {r}")
