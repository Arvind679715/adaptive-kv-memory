"""Multi-model evaluation sweep for AKV / NormQuant.

Runs perplexity + passkey retrieval across multiple model architectures
to validate generalization beyond TinyLlama / Llama-2-7B.

Supported models:
  - meta-llama/Meta-Llama-3-8B
  - Qwen/Qwen2.5-7B
  - mistralai/Mistral-7B-v0.3
  - google/gemma-2-9b
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (baseline/sanity)

Usage:
    python -m benchmarks.model_sweep --models llama3,qwen2.5,mistral
    python -m benchmarks.model_sweep --all --output-dir ./sweep_results
    python -m benchmarks.model_sweep --models llama3 --eval ppl,passkey
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-inst": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "gemma2": "google/gemma-2-9b",
}


@dataclass
class SweepConfig:
    """Configuration for the model sweep."""
    models: list[str]
    evaluations: list[str]  # "ppl", "passkey", "needle"
    output_dir: str = "./sweep_results"
    # Perplexity settings
    ppl_max_tokens: int = 4096
    ppl_stride: int = 512
    # Passkey settings
    passkey_lengths: list[int] = None
    # Cache configs to compare
    hot_budget: int = 512
    warm_budget: int = 2048
    warm_bits: int = 3
    device: str = "cuda"
    dtype: str = "float16"

    def __post_init__(self):
        if self.passkey_lengths is None:
            self.passkey_lengths = [1024, 2048, 4096]


def resolve_model_names(model_keys: list[str]) -> list[tuple[str, str]]:
    """Resolve short model keys to (key, hf_model_id) pairs."""
    results = []
    for key in model_keys:
        key_lower = key.lower().strip()
        if key_lower in MODEL_REGISTRY:
            results.append((key_lower, MODEL_REGISTRY[key_lower]))
        else:
            # Treat as raw HF model ID
            short = key_lower.split("/")[-1][:20]
            results.append((short, key))
    return results


def run_perplexity_eval(model_name: str, config: SweepConfig) -> dict:
    """Run perplexity evaluation for a single model with AKV vs baselines."""
    from akv.evaluation import EvalConfig

    logger.info(f"[PPL] Evaluating {model_name}")
    eval_config = EvalConfig(
        model_name=model_name,
        dataset="wikitext",
        max_eval_tokens=config.ppl_max_tokens,
        stride=config.ppl_stride,
        device=config.device,
        dtype=config.dtype,
    )

    results = {}

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Use 4-bit quantization to fit 7B models on single T4 (16GB)
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit quantization (BitsAndBytesConfig)")
        except ImportError:
            logger.info("bitsandbytes not available, using FP16")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        # Load WikiText-2
        from datasets import load_dataset
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in dataset["text"] if t.strip())
        encodings = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=config.ppl_max_tokens)
        input_ids = encodings.input_ids.to(config.device)

        # Baseline: full cache perplexity
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            results["full_cache_ppl"] = outputs.loss.exp().item()

        # AKV with NormQuant
        from akv.integration import HFProductionCache
        from akv.production_cache import ProductionCacheConfig

        # Get model config
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

        # Evaluate with production cache
        cache = HFProductionCache(prod_config)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, past_key_values=cache)
            results["akv_ppl"] = outputs.loss.exp().item()

        results["ppl_degradation_pct"] = (
            (results["akv_ppl"] - results["full_cache_ppl"]) / results["full_cache_ppl"] * 100
        )
        results["status"] = "success"

    except Exception as e:
        logger.error(f"[PPL] Failed for {model_name}: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_passkey_eval(model_name: str, config: SweepConfig) -> dict:
    """Run passkey retrieval evaluation for a single model."""
    import random

    logger.info(f"[Passkey] Evaluating {model_name}")
    results = {}

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from akv.integration import HFProductionCache
        from akv.production_cache import ProductionCacheConfig

        # Use 4-bit quantization to fit 7B models on single T4 (16GB)
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            pass

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        num_layers = model.config.num_hidden_layers
        num_heads = getattr(model.config, "num_key_value_heads",
                           model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        filler_sentences = [
            "The grass is green. The sky is blue. The sun is yellow.",
            "Technology advances rapidly in the modern era of computing.",
            "Mountains rise above the plains, touching the clouds.",
            "Rivers flow from the highlands to the sea.",
            "Cities grow and evolve, shaped by the people who inhabit them.",
            "Science explores the boundaries of human understanding.",
            "Music fills the air with melodies that resonate across cultures.",
            "The ocean is vast, covering most of the surface of our planet.",
        ]

        for ctx_len in config.passkey_lengths:
            key = f"ctx_{ctx_len}"
            correct = 0
            num_trials = 5

            try:
                for trial in range(num_trials):
                    passkey = str(random.randint(10000, 99999))
                    needle = f"The secret passkey is: {passkey}. Remember this number."

                    # Build context with needle at ~25% depth
                    target_chars = ctx_len * 4  # rough char estimate
                    filler_block = " ".join(filler_sentences)
                    repeats = max(1, target_chars // len(filler_block))

                    insert_pos = repeats // 4
                    parts = [filler_block] * insert_pos + [needle] + [filler_block] * (repeats - insert_pos)
                    context = " ".join(parts)

                    prompt = context + "\n\nWhat is the secret passkey mentioned above? The passkey is:"

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                      max_length=ctx_len).to(config.device)

                    # Generate with AKV cache
                    prod_config = ProductionCacheConfig(
                        num_layers=num_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        hot_budget=config.hot_budget,
                        warm_budget=config.warm_budget,
                        warm_bits=config.warm_bits,
                    )
                    cache = HFProductionCache(prod_config)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            past_key_values=cache,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

                    if passkey in answer:
                        correct += 1

                results[key] = {
                    "accuracy": correct / num_trials,
                    "correct": correct,
                    "total": num_trials,
                    "context_length": ctx_len,
                }
            except Exception as e:
                results[key] = {"status": "error", "error": str(e)}

        results["status"] = "success"

    except Exception as e:
        logger.error(f"[Passkey] Failed for {model_name}: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_sweep(config: SweepConfig) -> dict:
    """Run the full multi-model sweep."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_pairs = resolve_model_names(config.models)
    all_results = {}

    for model_key, model_id in model_pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_id} ({model_key})")
        logger.info(f"{'='*60}")

        model_results = {"model_id": model_id, "model_key": model_key}

        if "ppl" in config.evaluations:
            model_results["perplexity"] = run_perplexity_eval(model_id, config)

        if "passkey" in config.evaluations:
            model_results["passkey"] = run_passkey_eval(model_id, config)

        all_results[model_key] = model_results

        # Save incremental results
        results_file = output_dir / "sweep_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")

    # Print summary table
    print_summary(all_results)
    return all_results


def print_summary(results: dict):
    """Print a summary table of the sweep results."""
    print("\n" + "=" * 70)
    print("MODEL SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Full PPL':>10} {'AKV PPL':>10} {'Degrad%':>8} {'Passkey':>10}")
    print("-" * 70)

    for key, data in results.items():
        model_short = key[:18]
        ppl_data = data.get("perplexity", {})
        passkey_data = data.get("passkey", {})

        full_ppl = ppl_data.get("full_cache_ppl", "N/A")
        akv_ppl = ppl_data.get("akv_ppl", "N/A")
        degrad = ppl_data.get("ppl_degradation_pct", "N/A")

        # Average passkey accuracy across context lengths
        passkey_accs = []
        for k, v in passkey_data.items():
            if k.startswith("ctx_") and isinstance(v, dict):
                acc = v.get("accuracy", v.get("recall", None))
                if acc is not None:
                    passkey_accs.append(acc)
        avg_passkey = f"{sum(passkey_accs)/len(passkey_accs)*100:.1f}%" if passkey_accs else "N/A"

        full_str = f"{full_ppl:.2f}" if isinstance(full_ppl, float) else str(full_ppl)
        akv_str = f"{akv_ppl:.2f}" if isinstance(akv_ppl, float) else str(akv_ppl)
        deg_str = f"{degrad:.1f}%" if isinstance(degrad, float) else str(degrad)

        print(f"{model_short:<20} {full_str:>10} {akv_str:>10} {deg_str:>8} {avg_passkey:>10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-model AKV evaluation sweep")
    parser.add_argument(
        "--models", type=str, default="tinyllama,llama3,qwen2.5",
        help="Comma-separated model keys or HF model IDs. "
             f"Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument("--all", action="store_true", help="Run all registered models")
    parser.add_argument(
        "--eval", type=str, default="ppl,passkey",
        help="Comma-separated evaluations: ppl, passkey"
    )
    parser.add_argument("--output-dir", type=str, default="./sweep_results")
    parser.add_argument("--hot-budget", type=int, default=512)
    parser.add_argument("--warm-budget", type=int, default=2048)
    parser.add_argument("--warm-bits", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all:
        models = list(MODEL_REGISTRY.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]

    config = SweepConfig(
        models=models,
        evaluations=[e.strip() for e in args.eval.split(",")],
        output_dir=args.output_dir,
        ppl_max_tokens=args.max_tokens,
        hot_budget=args.hot_budget,
        warm_budget=args.warm_budget,
        warm_bits=args.warm_bits,
        device=args.device,
    )

    logger.info(f"Starting model sweep: {config.models}")
    logger.info(f"Evaluations: {config.evaluations}")
    run_sweep(config)


if __name__ == "__main__":
    main()
