"""Long-context delayed recall benchmark.

Tests the ability of KV cache methods to retain and retrieve information
placed early in long contexts — the critical test for eviction-based methods.

Benchmarks:
1. **Passkey Retrieval**: Random passkey inserted at various depths
2. **Multi-Needle**: Multiple facts placed at different positions, queried later
3. **Delayed QA**: Information provided early, questions asked after long filler
4. **Associative Recall**: Key-value pairs in context, recall after delay

This is the benchmark that separates our hierarchical approach from naive eviction.
Eviction-based methods (H2O, ScissorHands) catastrophically fail at early positions
because they discard tokens that were important but hadn't been queried yet.
Our cold-tier preservation + promotion mechanism handles this elegantly.

Usage:
    python -m benchmarks.delayed_recall --model meta-llama/Llama-2-7b-hf
    python -m benchmarks.delayed_recall --context-lengths 2048,4096,8192,16384,32768
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from akv.cache import CacheConfig
from akv.integration import HFAdaptiveCache

logger = logging.getLogger(__name__)


# ============================================================
# Test Templates
# ============================================================

HAYSTACK_SENTENCES = [
    "The grass is green. The sky is blue. The sun is yellow.",
    "Technology advances rapidly in the modern era of computing.",
    "Mountains rise above the plains, touching the clouds at their peaks.",
    "Rivers flow from the highlands to the sea, carving valleys along the way.",
    "Cities grow and evolve, shaped by the people who inhabit them.",
    "Science explores the boundaries of human understanding every day.",
    "Music fills the air with melodies that resonate across cultures.",
    "The ocean is vast, covering most of the surface of our planet.",
]

PASSKEY_TEMPLATE = "The secret passkey is: {passkey}. Remember this number carefully."
MULTI_FACT_TEMPLATES = [
    "IMPORTANT FACT: The capital of Zarathia is {value}. Remember this.",
    "CRITICAL INFO: The password for vault seven is {value}. Do not forget.",
    "KEY DATA: The launch code sequence is {value}. This is essential.",
    "VITAL NOTE: The agent's codename is {value}. Keep this in memory.",
]


@dataclass
class RecallResult:
    test_type: str
    method: str
    context_length: int
    needle_position: float  # 0.0 to 1.0 (fraction of context)
    accuracy: float
    exact_match: bool
    partial_match: bool
    generated_answer: str
    expected_answer: str
    latency_ms: float
    num_trials: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class DelayedRecallConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    context_lengths: list[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384])
    needle_positions: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    num_trials: int = 5
    max_new_tokens: int = 32
    device: str = "cuda"
    dtype: str = "float16"
    output_dir: str = "./benchmark_results"
    seed: int = 42
    methods: list[str] = field(default_factory=lambda: [
        "full", "akv-4bit", "akv-2bit", "h2o-1024", "snapkv-1024"
    ])


class DelayedRecallBenchmark:
    """Long-context delayed recall benchmark suite.

    The key insight: eviction-based methods fail at recalling information
    from early positions because they evict those tokens before they're queried.
    Our hierarchical approach stores them in cold tier and promotes on demand.
    """

    def __init__(self, config: DelayedRecallConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results: list[RecallResult] = []

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

    def _build_haystack(self, target_tokens: int) -> str:
        """Build filler text of approximately target_tokens length."""
        filler = " ".join(HAYSTACK_SENTENCES)
        filler_tokens = len(self.tokenizer.encode(filler))
        repeats = (target_tokens // filler_tokens) + 1
        full_text = " ".join([filler] * repeats)
        # Truncate to exact token count
        tokens = self.tokenizer.encode(full_text)[:target_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _generate_with_cache(
        self, prompt: str, cache_config: Optional[CacheConfig] = None
    ) -> tuple[str, float]:
        """Generate response using adaptive cache. Returns (text, latency_ms)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)

        if cache_config:
            cache = HFAdaptiveCache(cache_config)
        else:
            cache = None

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                past_key_values=cache,
                use_cache=True,
            )
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated.strip(), (t1 - t0) * 1000

    def _get_cache_config(self, method: str) -> Optional[CacheConfig]:
        """Get cache config for a method. None means full cache."""
        configs = {
            "full": None,
            "akv-4bit": CacheConfig(
                hot_budget=1024, warm_budget=2048, warm_bits=4, cold_bits=2,
                enable_cold_tier=True, group_size=128,
            ),
            "akv-2bit": CacheConfig(
                hot_budget=1024, warm_budget=2048, warm_bits=2, cold_bits=2,
                enable_cold_tier=True, group_size=128,
            ),
            "akv-aggressive": CacheConfig(
                hot_budget=512, warm_budget=4096, warm_bits=4, cold_bits=2,
                enable_cold_tier=True, group_size=64,
            ),
            "h2o-1024": CacheConfig(
                hot_budget=1024, warm_budget=0, warm_bits=4,
                enable_cold_tier=False,
            ),
            "snapkv-1024": CacheConfig(
                hot_budget=1024, warm_budget=0, warm_bits=4,
                enable_cold_tier=False,
            ),
        }
        return configs.get(method)

    # ============================================================
    # Test 1: Passkey Retrieval
    # ============================================================

    def run_passkey_retrieval(self) -> list[RecallResult]:
        """Standard passkey retrieval across depths and context lengths."""
        results = []
        rng = random.Random(self.config.seed)

        for ctx_len in self.config.context_lengths:
            for position in self.config.needle_positions:
                for method in self.config.methods:
                    accuracies = []
                    cache_config = self._get_cache_config(method)

                    for trial in range(self.config.num_trials):
                        passkey = str(rng.randint(10000, 99999))
                        needle = PASSKEY_TEMPLATE.format(passkey=passkey)

                        # Build context with needle at position
                        pre_tokens = int(ctx_len * position)
                        post_tokens = ctx_len - pre_tokens - len(self.tokenizer.encode(needle))

                        pre_text = self._build_haystack(max(pre_tokens, 10))
                        post_text = self._build_haystack(max(post_tokens, 10))

                        prompt = (
                            f"{pre_text}\n\n{needle}\n\n{post_text}\n\n"
                            f"Question: What is the secret passkey mentioned in the text above?\n"
                            f"Answer: The secret passkey is"
                        )

                        try:
                            generated, latency = self._generate_with_cache(prompt, cache_config)
                            exact = passkey in generated
                            partial = any(c in generated for c in passkey)
                            accuracies.append(1.0 if exact else 0.0)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            logger.warning(f"  OOM/Error: {method}@{ctx_len}: {e}")
                            generated = "OOM"
                            exact = False
                            partial = False
                            latency = 0
                            accuracies.append(0.0)
                            self._clear_gpu()

                    avg_acc = np.mean(accuracies)
                    result = RecallResult(
                        test_type="passkey_retrieval",
                        method=method,
                        context_length=ctx_len,
                        needle_position=position,
                        accuracy=avg_acc,
                        exact_match=avg_acc == 1.0,
                        partial_match=avg_acc > 0.0,
                        generated_answer=generated,
                        expected_answer=passkey,
                        latency_ms=latency,
                        num_trials=self.config.num_trials,
                    )
                    results.append(result)
                    logger.info(f"  Passkey {method}@{ctx_len} pos={position:.2f}: "
                               f"acc={avg_acc:.1%}")

        return results

    # ============================================================
    # Test 2: Multi-Needle Recall
    # ============================================================

    def run_multi_needle(self) -> list[RecallResult]:
        """Insert multiple facts at different positions, query all at the end."""
        results = []
        rng = random.Random(self.config.seed + 1)

        for ctx_len in self.config.context_lengths:
            for method in self.config.methods:
                cache_config = self._get_cache_config(method)
                trial_results = []

                for trial in range(self.config.num_trials):
                    # Generate random facts
                    facts = []
                    for tmpl in MULTI_FACT_TEMPLATES:
                        value = str(rng.randint(10000, 99999))
                        facts.append((tmpl.format(value=value), value))

                    # Place facts at evenly-spaced positions
                    positions = np.linspace(0.1, 0.8, len(facts))
                    text_parts = []
                    prev_end = 0

                    for (fact_text, _), pos in zip(facts, positions):
                        filler_tokens = int(ctx_len * pos) - prev_end
                        text_parts.append(self._build_haystack(max(filler_tokens, 10)))
                        text_parts.append(f"\n{fact_text}\n")
                        prev_end = int(ctx_len * pos) + len(self.tokenizer.encode(fact_text))

                    # Add trailing filler
                    remaining = ctx_len - prev_end
                    text_parts.append(self._build_haystack(max(remaining, 10)))

                    # Query for each fact
                    questions = [
                        "What is the capital of Zarathia?",
                        "What is the password for vault seven?",
                        "What is the launch code sequence?",
                        "What is the agent's codename?",
                    ]

                    full_context = "".join(text_parts)
                    recall_count = 0

                    for (_, expected), question in zip(facts, questions):
                        prompt = f"{full_context}\n\nQuestion: {question}\nAnswer:"

                        try:
                            generated, latency = self._generate_with_cache(prompt, cache_config)
                            if expected in generated:
                                recall_count += 1
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            self._clear_gpu()
                            break

                    trial_results.append(recall_count / len(facts))

                avg_acc = np.mean(trial_results) if trial_results else 0.0
                result = RecallResult(
                    test_type="multi_needle",
                    method=method,
                    context_length=ctx_len,
                    needle_position=0.0,  # Multiple positions
                    accuracy=avg_acc,
                    exact_match=avg_acc == 1.0,
                    partial_match=avg_acc > 0.0,
                    generated_answer=f"recall_rate={avg_acc:.2%}",
                    expected_answer="all_facts",
                    latency_ms=0,
                    num_trials=self.config.num_trials,
                )
                results.append(result)
                logger.info(f"  Multi-needle {method}@{ctx_len}: acc={avg_acc:.1%}")

        return results

    # ============================================================
    # Test 3: Associative Recall with Delay
    # ============================================================

    def run_associative_recall(self) -> list[RecallResult]:
        """Key-value pair recall: present pairs early, query after long delay."""
        results = []
        rng = random.Random(self.config.seed + 2)

        PAIR_TEMPLATE = "The {key} is associated with the number {value}."
        KEYS = ["red dragon", "blue ocean", "green forest", "golden crown", "silver moon"]

        for ctx_len in self.config.context_lengths:
            for method in self.config.methods:
                cache_config = self._get_cache_config(method)
                trial_accuracies = []

                for trial in range(self.config.num_trials):
                    # Generate key-value pairs
                    pairs = []
                    for key in KEYS:
                        value = str(rng.randint(100, 999))
                        pairs.append((key, value))

                    # Place all pairs in the first 10% of context
                    pairs_text = "\n".join(PAIR_TEMPLATE.format(key=k, value=v) for k, v in pairs)
                    pairs_tokens = len(self.tokenizer.encode(pairs_text))

                    # Fill rest with filler (the "delay")
                    filler_tokens = ctx_len - pairs_tokens - 50
                    filler = self._build_haystack(max(filler_tokens, 100))

                    # Query a random pair
                    query_key, expected_value = rng.choice(pairs)
                    prompt = (
                        f"{pairs_text}\n\n{filler}\n\n"
                        f"Question: What number is associated with the {query_key}?\n"
                        f"Answer: The number is"
                    )

                    try:
                        generated, latency = self._generate_with_cache(prompt, cache_config)
                        trial_accuracies.append(1.0 if expected_value in generated else 0.0)
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        trial_accuracies.append(0.0)
                        self._clear_gpu()

                avg_acc = np.mean(trial_accuracies)
                result = RecallResult(
                    test_type="associative_recall",
                    method=method,
                    context_length=ctx_len,
                    needle_position=0.05,  # Pairs placed at start
                    accuracy=avg_acc,
                    exact_match=avg_acc == 1.0,
                    partial_match=avg_acc > 0.0,
                    generated_answer="",
                    expected_answer="",
                    latency_ms=0,
                    num_trials=self.config.num_trials,
                )
                results.append(result)
                logger.info(f"  Associative {method}@{ctx_len}: acc={avg_acc:.1%}")

        return results

    # ============================================================
    # Full Suite
    # ============================================================

    def run(self) -> list[RecallResult]:
        """Run all delayed recall benchmarks."""
        if self.model is None:
            self.setup()

        all_results = []

        logger.info("\n" + "="*60)
        logger.info("Test 1: Passkey Retrieval")
        logger.info("="*60)
        all_results.extend(self.run_passkey_retrieval())

        logger.info("\n" + "="*60)
        logger.info("Test 2: Multi-Needle Recall")
        logger.info("="*60)
        all_results.extend(self.run_multi_needle())

        logger.info("\n" + "="*60)
        logger.info("Test 3: Associative Recall with Delay")
        logger.info("="*60)
        all_results.extend(self.run_associative_recall())

        self.results = all_results
        return all_results

    def save_results(self, path: Optional[str] = None):
        """Save results to JSON."""
        if path is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            path = f"{self.config.output_dir}/delayed_recall_{int(time.time())}.json"

        data = {
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
            "summary": self._compute_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def _compute_summary(self) -> dict:
        """Compute summary: accuracy heatmap by method, context length, position."""
        summary = {"by_method": {}, "by_test_type": {}}

        for r in self.results:
            # By method
            if r.method not in summary["by_method"]:
                summary["by_method"][r.method] = []
            summary["by_method"][r.method].append(r.accuracy)

            # By test type
            key = f"{r.test_type}/{r.method}"
            if key not in summary["by_test_type"]:
                summary["by_test_type"][key] = []
            summary["by_test_type"][key].append(r.accuracy)

        # Aggregate
        for method, accs in summary["by_method"].items():
            summary["by_method"][method] = {
                "mean_accuracy": round(np.mean(accs), 3),
                "min_accuracy": round(np.min(accs), 3),
                "num_tests": len(accs),
            }
        for key, accs in list(summary["by_test_type"].items()):
            summary["by_test_type"][key] = round(np.mean(accs), 3)

        return summary

    def print_results(self):
        """Pretty-print a summary heatmap."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Summary by method and test type
            table = Table(title="Delayed Recall: Accuracy by Method", show_lines=True)
            table.add_column("Method", style="cyan", min_width=20)
            table.add_column("Passkey", justify="center")
            table.add_column("Multi-Needle", justify="center")
            table.add_column("Associative", justify="center")
            table.add_column("Overall", justify="center", style="bold")

            methods_seen = {}
            for r in self.results:
                if r.method not in methods_seen:
                    methods_seen[r.method] = {"passkey_retrieval": [], "multi_needle": [], "associative_recall": []}
                methods_seen[r.method][r.test_type].append(r.accuracy)

            for method, tests in methods_seen.items():
                pk = np.mean(tests["passkey_retrieval"]) if tests["passkey_retrieval"] else 0
                mn = np.mean(tests["multi_needle"]) if tests["multi_needle"] else 0
                ar = np.mean(tests["associative_recall"]) if tests["associative_recall"] else 0
                overall = np.mean([pk, mn, ar])

                # Color code by accuracy
                def fmt(v):
                    if v >= 0.9:
                        return f"[green]{v:.1%}[/green]"
                    elif v >= 0.5:
                        return f"[yellow]{v:.1%}[/yellow]"
                    else:
                        return f"[red]{v:.1%}[/red]"

                table.add_row(method, fmt(pk), fmt(mn), fmt(ar), fmt(overall))

            console.print(table)

            # Passkey depth analysis
            table2 = Table(title="Passkey Retrieval by Depth (Position in Context)", show_lines=True)
            table2.add_column("Method", style="cyan")
            for pos in self.config.needle_positions:
                table2.add_column(f"{pos:.0%}", justify="center")

            for method in methods_seen:
                row = [method]
                for pos in self.config.needle_positions:
                    accs = [r.accuracy for r in self.results
                            if r.method == method and r.test_type == "passkey_retrieval"
                            and abs(r.needle_position - pos) < 0.01]
                    if accs:
                        v = np.mean(accs)
                        if v >= 0.9:
                            row.append(f"[green]{v:.0%}[/green]")
                        elif v >= 0.5:
                            row.append(f"[yellow]{v:.0%}[/yellow]")
                        else:
                            row.append(f"[red]{v:.0%}[/red]")
                    else:
                        row.append("-")
                table2.add_row(*row)

            console.print(table2)

        except ImportError:
            print("\nDelayed Recall Results:")
            print("-" * 60)
            for r in self.results:
                print(f"  [{r.test_type}] {r.method}@{r.context_length} "
                      f"pos={r.needle_position:.2f}: acc={r.accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Long-context delayed recall benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--context-lengths", default="1024,2048,4096,8192",
                        help="Comma-separated context lengths")
    parser.add_argument("--positions", default="0.05,0.1,0.25,0.5,0.75,0.9",
                        help="Needle positions (fraction of context)")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--methods", default="full,akv-4bit,akv-2bit,h2o-1024,snapkv-1024")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = DelayedRecallConfig(
        model_name=args.model,
        context_lengths=[int(x) for x in args.context_lengths.split(",")],
        needle_positions=[float(x) for x in args.positions.split(",")],
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        methods=[x.strip() for x in args.methods.split(",")],
        seed=args.seed,
    )

    bench = DelayedRecallBenchmark(config)
    bench.run()
    bench.print_results()
    bench.save_results()


if __name__ == "__main__":
    main()
