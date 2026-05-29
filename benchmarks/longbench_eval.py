"""LongBench evaluation harness for AKV cache methods.

Evaluates KV cache compression methods on LongBench tasks:
- Single-document QA (NarrativeQA, Qasper, MultiFieldQA)
- Multi-document QA (HotpotQA, 2WikiMQA, MuSiQue)
- Summarization (GovReport, QMSum, MultiNews)
- Few-shot learning (TREC, TriviaQA, SAMSum)
- Code completion (LCC, RepoBench-P)
- Synthetic (PassageCount, PassageRetrieval)

Usage:
    python -m benchmarks.longbench_eval --model meta-llama/Meta-Llama-3-8B
    python -m benchmarks.longbench_eval --tasks narrativeqa,hotpotqa --methods akv,full,h2o
    python -m benchmarks.longbench_eval --all-tasks --output-dir ./longbench_results
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ============================================================
# Task registry
# ============================================================

LONGBENCH_TASKS = {
    # Single-document QA
    "narrativeqa": {"category": "single_doc_qa", "metric": "f1"},
    "qasper": {"category": "single_doc_qa", "metric": "f1"},
    "multifieldqa_en": {"category": "single_doc_qa", "metric": "f1"},
    # Multi-document QA
    "hotpotqa": {"category": "multi_doc_qa", "metric": "f1"},
    "2wikimqa": {"category": "multi_doc_qa", "metric": "f1"},
    "musique": {"category": "multi_doc_qa", "metric": "f1"},
    # Summarization
    "gov_report": {"category": "summarization", "metric": "rouge"},
    "qmsum": {"category": "summarization", "metric": "rouge"},
    "multi_news": {"category": "summarization", "metric": "rouge"},
    # Few-shot
    "trec": {"category": "few_shot", "metric": "accuracy"},
    "triviaqa": {"category": "few_shot", "metric": "f1"},
    "samsum": {"category": "few_shot", "metric": "rouge"},
    # Code
    "lcc": {"category": "code", "metric": "edit_sim"},
    "repobench-p": {"category": "code", "metric": "edit_sim"},
    # Synthetic
    "passage_count": {"category": "synthetic", "metric": "accuracy"},
    "passage_retrieval_en": {"category": "synthetic", "metric": "accuracy"},
}

TASK_CATEGORIES = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "code": ["lcc", "repobench-p"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
}


@dataclass
class LongBenchConfig:
    """Configuration for LongBench evaluation."""
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    tasks: list[str] = field(default_factory=lambda: ["narrativeqa", "hotpotqa", "gov_report"])
    methods: list[str] = field(default_factory=lambda: ["akv", "full", "h2o"])
    max_length: int = 8192        # Max context length
    max_gen_tokens: int = 512     # Max generation tokens
    output_dir: str = "./longbench_results"
    # AKV settings
    hot_budget: int = 512
    warm_budget: int = 4096
    warm_bits: int = 3
    # H2O / baseline settings
    baseline_budget: int = 2048
    device: str = "cuda"
    dtype: str = "float16"
    max_samples: int = 50         # Max samples per task (for speed)


def load_longbench_task(task_name: str, max_samples: int = 50) -> list[dict]:
    """Load a LongBench task dataset from HuggingFace.

    Returns list of dicts with keys: input, context, answers, length.
    """
    from datasets import load_dataset

    # Load directly from JSONL files to avoid dataset script issues with datasets>=3.0
    url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{task_name}.jsonl"
    try:
        dataset = load_dataset("json", data_files=url, split="train")
    except Exception:
        url_alt = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{task_name.replace('-', '_')}.jsonl"
        dataset = load_dataset("json", data_files=url_alt, split="train")

    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        samples.append({
            "input": item.get("input", ""),
            "context": item.get("context", ""),
            "answers": item.get("answers", [item.get("answer", "")]),
            "length": item.get("length", 0),
        })

    return samples


def build_prompt(task_name: str, sample: dict) -> str:
    """Build the prompt for a LongBench task sample."""
    context = sample["context"]
    question = sample["input"]

    task_info = LONGBENCH_TASKS[task_name]
    category = task_info["category"]

    if category in ("single_doc_qa", "multi_doc_qa"):
        prompt = (
            f"Read the following text and answer the question.\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    elif category == "summarization":
        prompt = (
            f"Summarize the following text.\n\n"
            f"Text: {context}\n\n"
            f"Summary:"
        )
    elif category == "few_shot":
        prompt = f"{context}\n\n{question}\nAnswer:"
    elif category == "code":
        prompt = f"{context}\n{question}"
    elif category == "synthetic":
        prompt = f"{context}\n\n{question}\nAnswer:"
    else:
        prompt = f"{context}\n\n{question}\nAnswer:"

    return prompt


# ============================================================
# Metrics
# ============================================================

def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """Compute token-level F1 score."""
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    best_f1 = 0.0
    pred_tokens = _tokenize(prediction)

    for gt in ground_truths:
        gt_tokens = _tokenize(gt)
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_rouge_l(prediction: str, ground_truths: list[str]) -> float:
    """Compute ROUGE-L F1 score."""
    def _lcs_length(x: list[str], y: list[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    pred_tokens = prediction.lower().split()
    best_rouge = 0.0

    for gt in ground_truths:
        gt_tokens = gt.lower().split()
        if not pred_tokens or not gt_tokens:
            continue
        lcs = _lcs_length(pred_tokens, gt_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0
        recall = lcs / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_rouge = max(best_rouge, f1)

    return best_rouge


def compute_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """Exact match accuracy (case-insensitive, stripped)."""
    pred_clean = prediction.strip().lower()
    for gt in ground_truths:
        if gt.strip().lower() in pred_clean or pred_clean in gt.strip().lower():
            return 1.0
    return 0.0


def compute_edit_sim(prediction: str, ground_truths: list[str]) -> float:
    """Compute edit similarity (1 - normalized edit distance)."""
    def _edit_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = min(prev, dp[j], dp[j-1]) + 1
                prev = temp
        return dp[n]

    best_sim = 0.0
    for gt in ground_truths:
        max_len = max(len(prediction), len(gt), 1)
        dist = _edit_distance(prediction[:500], gt[:500])  # Cap length for speed
        sim = 1.0 - dist / max_len
        best_sim = max(best_sim, sim)

    return best_sim


def score_sample(task_name: str, prediction: str, ground_truths: list[str]) -> float:
    """Score a prediction against ground truths using the task's metric."""
    metric = LONGBENCH_TASKS[task_name]["metric"]
    if metric == "f1":
        return compute_f1(prediction, ground_truths)
    elif metric == "rouge":
        return compute_rouge_l(prediction, ground_truths)
    elif metric == "accuracy":
        return compute_accuracy(prediction, ground_truths)
    elif metric == "edit_sim":
        return compute_edit_sim(prediction, ground_truths)
    else:
        return compute_f1(prediction, ground_truths)


# ============================================================
# Evaluation runner
# ============================================================

def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    method: str,
    config: LongBenchConfig,
) -> str:
    """Generate text using specified cache method."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
    ).to(config.device)

    past_key_values = None

    if method == "akv":
        from akv.integration import HFProductionCache
        from akv.production_cache import ProductionCacheConfig

        num_layers = model.config.num_hidden_layers
        num_heads = getattr(model.config, "num_key_value_heads",
                           model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        prod_config = ProductionCacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hot_budget=config.hot_budget,
            warm_budget=config.warm_budget,
            warm_bits=config.warm_bits,
        )
        past_key_values = HFProductionCache(prod_config)

    elif method == "h2o":
        from akv.integration import HFAdaptiveCache
        from akv.cache import CacheConfig
        cache_config = CacheConfig(
            budget=config.baseline_budget,
            strategy="HYBRID",
        )
        past_key_values = HFAdaptiveCache(cache_config)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_gen_tokens,
            past_key_values=past_key_values,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return prediction.strip()


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    method: str,
    config: LongBenchConfig,
) -> dict:
    """Evaluate a single task with a single method."""
    logger.info(f"  [{method}] Task: {task_name}")

    samples = load_longbench_task(task_name, max_samples=config.max_samples)
    scores = []
    errors = 0

    for i, sample in enumerate(samples):
        prompt = build_prompt(task_name, sample)

        try:
            prediction = generate_with_cache(model, tokenizer, prompt, method, config)
            score = score_sample(task_name, prediction, sample["answers"])
            scores.append(score)
        except Exception as e:
            logger.warning(f"    Sample {i} failed: {e}")
            errors += 1
            scores.append(0.0)

        if (i + 1) % 10 == 0:
            logger.info(f"    Progress: {i+1}/{len(samples)}, avg={sum(scores)/len(scores):.3f}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "task": task_name,
        "method": method,
        "metric": LONGBENCH_TASKS[task_name]["metric"],
        "avg_score": avg_score,
        "num_samples": len(samples),
        "num_errors": errors,
        "scores": scores,
    }


def run_longbench(config: LongBenchConfig) -> dict:
    """Run the full LongBench evaluation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {
        "model": config.model_name,
        "config": {
            "hot_budget": config.hot_budget,
            "warm_budget": config.warm_budget,
            "warm_bits": config.warm_bits,
            "baseline_budget": config.baseline_budget,
            "max_length": config.max_length,
        },
        "tasks": {},
    }

    for task_name in config.tasks:
        if task_name not in LONGBENCH_TASKS:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Task: {task_name} ({LONGBENCH_TASKS[task_name]['category']})")
        logger.info(f"{'='*50}")

        task_results = {}
        for method in config.methods:
            result = evaluate_task(model, tokenizer, task_name, method, config)
            task_results[method] = result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results["tasks"][task_name] = task_results

        # Save incremental
        results_file = output_dir / "longbench_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print_longbench_summary(all_results)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


def print_longbench_summary(results: dict):
    """Print a summary table of LongBench results."""
    methods = list(set(
        m for task_data in results["tasks"].values() for m in task_data.keys()
    ))
    methods.sort()

    print("\n" + "=" * 70)
    print(f"LONGBENCH RESULTS — {results['model']}")
    print("=" * 70)

    header = f"{'Task':<25} {'Metric':<8}"
    for m in methods:
        header += f" {m:>8}"
    print(header)
    print("-" * 70)

    category_scores = {}  # {category: {method: [scores]}}

    for task_name, task_data in results["tasks"].items():
        metric = LONGBENCH_TASKS.get(task_name, {}).get("metric", "?")
        cat = LONGBENCH_TASKS.get(task_name, {}).get("category", "?")

        row = f"{task_name:<25} {metric:<8}"
        for m in methods:
            if m in task_data:
                score = task_data[m]["avg_score"]
                row += f" {score:>8.3f}"

                if cat not in category_scores:
                    category_scores[cat] = {}
                if m not in category_scores[cat]:
                    category_scores[cat][m] = []
                category_scores[cat][m].append(score)
            else:
                row += f" {'N/A':>8}"
        print(row)

    # Category averages
    print("-" * 70)
    print("Category Averages:")
    for cat, method_scores in sorted(category_scores.items()):
        row = f"  {cat:<23}"
        for m in methods:
            if m in method_scores:
                avg = sum(method_scores[m]) / len(method_scores[m])
                row += f" {avg:>8.3f}"
            else:
                row += f" {'N/A':>8}"
        print(row)

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LongBench evaluation for AKV cache methods")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--tasks", type=str, default="narrativeqa,hotpotqa,gov_report",
        help=f"Comma-separated tasks. Available: {', '.join(LONGBENCH_TASKS.keys())}"
    )
    parser.add_argument("--all-tasks", action="store_true", help="Run all LongBench tasks")
    parser.add_argument(
        "--methods", type=str, default="akv,full,h2o",
        help="Comma-separated methods: akv, full, h2o"
    )
    parser.add_argument("--output-dir", type=str, default="./longbench_results")
    parser.add_argument("--hot-budget", type=int, default=512)
    parser.add_argument("--warm-budget", type=int, default=4096)
    parser.add_argument("--warm-bits", type=int, default=3)
    parser.add_argument("--baseline-budget", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all_tasks:
        tasks = list(LONGBENCH_TASKS.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    methods = [m.strip() for m in args.methods.split(",")]

    config = LongBenchConfig(
        model_name=args.model,
        tasks=tasks,
        methods=methods,
        output_dir=args.output_dir,
        hot_budget=args.hot_budget,
        warm_budget=args.warm_budget,
        warm_bits=args.warm_bits,
        baseline_budget=args.baseline_budget,
        max_length=args.max_length,
        max_samples=args.max_samples,
        device=args.device,
    )

    logger.info(f"LongBench evaluation: {config.model_name}")
    logger.info(f"Tasks: {config.tasks}")
    logger.info(f"Methods: {config.methods}")
    run_longbench(config)


if __name__ == "__main__":
    main()
