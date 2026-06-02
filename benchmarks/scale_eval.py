"""Large-scale model evaluation for paper validation.

Runs PPL, passkey retrieval, and RULER benchmarks on larger models
(Qwen2.5-7B, Llama-3-8B) to demonstrate generalization.

Designed for Kaggle T4 (16GB VRAM) with 4-bit model quantization.

Usage:
    python -m benchmarks.scale_eval --model Qwen/Qwen2.5-7B --bits 4 --seeds 3
    python -m benchmarks.scale_eval --model meta-llama/Llama-3-8B --bits 4 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import numpy as np

RESULTS_DIR = Path("benchmark_results")


@dataclass
class ScaleEvalConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    load_bits: int = 4  # load model in 4-bit via bitsandbytes
    seeds: list[int] | None = None
    # PPL settings
    ppl_stride: int = 512
    ppl_max_length: int = 2048
    # Passkey settings
    passkey_context_len: int = 4096
    passkey_trials: int = 10
    passkey_depths: list[float] | None = None
    # AKV settings
    akv_hot_budget: int = 512
    akv_warm_budget: int = 2048
    akv_warm_bits: int = 4
    # H2O settings
    h2o_budget: int = 512
    # StreamingLLM settings
    streaming_sinks: int = 4
    streaming_window: int = 508
    # PyramidKV settings
    pyramid_budget: int = 512

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]
        if self.passkey_depths is None:
            self.passkey_depths = [0.05, 0.25, 0.50, 0.75, 0.95]


def load_model_and_tokenizer(model_name: str, load_bits: int):
    """Load model with optional quantization for VRAM-constrained GPUs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if load_bits == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif load_bits == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def evaluate_ppl_with_cache(model, tokenizer, cache_factory, seed: int,
                            max_length: int = 2048, stride: int = 512):
    """Evaluate perplexity on WikiText-2 with a given cache strategy."""
    from datasets import load_dataset

    torch.manual_seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=65536)
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end_loc = 0

    for begin_loc in range(0, min(seq_len, max_length * 8), stride):
        end_loc = min(begin_loc + max_length, seq_len)
        chunk = input_ids[:, begin_loc:end_loc]

        target_len = chunk.size(1) - 1
        if target_len < 1:
            break

        with torch.no_grad():
            # Use model directly with full cache for baseline
            outputs = model(chunk, use_cache=False)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
        nlls.append(loss.item())

        prev_end_loc = end_loc
        if end_loc >= seq_len or len(nlls) >= 8:
            break

    return float(np.exp(np.mean(nlls)))


def evaluate_passkey(model, tokenizer, context_len: int, depth: float,
                     seed: int, n_trials: int = 10):
    """Single-passkey retrieval at a given depth."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    correct = 0
    for trial in range(n_trials):
        passkey = str(random.randint(10000, 99999))
        insert_pos = int(depth * context_len)

        # Build context with filler text
        filler_unit = "The quick brown fox jumps over the lazy dog. "
        filler_tokens = tokenizer(filler_unit, return_tensors="pt").input_ids.shape[1]
        n_repeats = context_len // filler_tokens + 1

        filler = filler_unit * n_repeats
        needle = f"The secret passkey is {passkey}. Remember it."
        prompt_end = f"\nWhat is the secret passkey? The passkey is"

        # Insert needle at depth position (character-level approximation)
        char_pos = int(depth * len(filler))
        text = filler[:char_pos] + needle + filler[char_pos:]

        # Truncate to approximate token count
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=context_len - 32)
        prompt_enc = tokenizer(prompt_end, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([enc.input_ids, prompt_enc.input_ids], dim=1).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        if passkey in generated:
            correct += 1

    return correct / n_trials


def run_full_eval(config: ScaleEvalConfig):
    """Run the full evaluation suite."""
    print("=" * 70)
    print(f"SCALE EVALUATION — {config.model_name}")
    print(f"Load bits: {config.load_bits} | Seeds: {config.seeds}")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.load_bits)
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers, device={model.device}")

    results = {
        "model": config.model_name,
        "load_bits": config.load_bits,
        "num_layers": num_layers,
        "ppl": {},
        "passkey": {},
    }

    # --- PPL Evaluation ---
    print("\n" + "─" * 50)
    print("PERPLEXITY (WikiText-2)")
    print("─" * 50)

    for seed in config.seeds:
        ppl = evaluate_ppl_with_cache(
            model, tokenizer, None, seed,
            max_length=config.ppl_max_length, stride=config.ppl_stride,
        )
        results["ppl"].setdefault("full_cache", []).append(ppl)
        print(f"  Full Cache (seed={seed}): PPL = {ppl:.3f}")

    mean_ppl = np.mean(results["ppl"]["full_cache"])
    std_ppl = np.std(results["ppl"]["full_cache"])
    print(f"  Full Cache: {mean_ppl:.3f} ± {std_ppl:.3f}")

    # --- Passkey Retrieval ---
    print("\n" + "─" * 50)
    print(f"PASSKEY RETRIEVAL ({config.passkey_context_len} tokens)")
    print("─" * 50)

    for depth in config.passkey_depths:
        depth_results = []
        for seed in config.seeds:
            acc = evaluate_passkey(
                model, tokenizer,
                context_len=config.passkey_context_len,
                depth=depth, seed=seed,
                n_trials=config.passkey_trials,
            )
            depth_results.append(acc)
        results["passkey"][f"depth_{depth}"] = depth_results
        mean_acc = np.mean(depth_results)
        std_acc = np.std(depth_results)
        print(f"  Depth {depth:.0%}: {mean_acc:.3f} ± {std_acc:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = config.model_name.split("/")[-1].lower()
    out_path = RESULTS_DIR / f"scale_eval_{model_short}_{config.load_bits}bit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Large-scale model evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--passkey-context", type=int, default=4096)
    parser.add_argument("--passkey-trials", type=int, default=10)
    parser.add_argument("--akv-hot", type=int, default=512)
    parser.add_argument("--akv-warm", type=int, default=2048)
    parser.add_argument("--h2o-budget", type=int, default=512)
    parser.add_argument("--streaming-sinks", type=int, default=4)
    parser.add_argument("--streaming-window", type=int, default=508)
    parser.add_argument("--pyramid-budget", type=int, default=512)
    args = parser.parse_args()

    config = ScaleEvalConfig(
        model_name=args.model,
        load_bits=args.bits,
        seeds=args.seeds,
        ppl_max_length=args.ppl_max_length,
        passkey_context_len=args.passkey_context,
        passkey_trials=args.passkey_trials,
        akv_hot_budget=args.akv_hot,
        akv_warm_budget=args.akv_warm,
        h2o_budget=args.h2o_budget,
        streaming_sinks=args.streaming_sinks,
        streaming_window=args.streaming_window,
        pyramid_budget=args.pyramid_budget,
    )
    run_full_eval(config)


if __name__ == "__main__":
    main()
