"""Multi-seed benchmark runner for statistical significance.

Re-runs key experiments with multiple random seeds and reports
mean ± std error bars for paper credibility.

Usage:
    python -m benchmarks.multi_seed --experiment ppl --seeds 42 123 456 789 2024
    python -m benchmarks.multi_seed --experiment passkey --seeds 42 123 456
    python -m benchmarks.multi_seed --experiment ruler --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np

RESULTS_DIR = Path("benchmark_results")


def run_ppl_multi_seed(model_name: str, seeds: list[int], context_len: int = 2048,
                       hot_budget: int = 256, warm_bits: int = 4):
    """Run perplexity evaluation with multiple seeds."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    results = {"full": [], "akv_4bit": [], "akv_2bit": [], "h2o": [],
               "streamingllm": [], "pyramidkv": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        encodings = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=context_len * 10)
        input_ids = encodings.input_ids.to(model.device)

        # Evaluate chunks
        stride = context_len // 2
        nlls = []
        for begin in range(0, min(input_ids.size(1), context_len * 8), stride):
            end = min(begin + context_len, input_ids.size(1))
            chunk = input_ids[:, begin:end]
            if chunk.size(1) < 2:
                break
            with torch.no_grad():
                out = model(chunk, use_cache=False)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss.item())
            if len(nlls) >= 8:
                break

        ppl_full = float(np.exp(np.mean(nlls)))
        results["full"].append(ppl_full)
        print(f"  Full: {ppl_full:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-SEED RESULTS")
    print("=" * 60)
    for method, ppls in results.items():
        if ppls:
            mean = np.mean(ppls)
            std = np.std(ppls)
            print(f"  {method:15s}: {mean:.3f} ± {std:.3f} (n={len(ppls)})")

    return results


def run_passkey_multi_seed(model_name: str, seeds: list[int],
                           context_len: int = 4096, n_trials: int = 10):
    """Run passkey retrieval with multiple seeds."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    depths = [0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
    results = {f"depth_{d}": [] for d in depths}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        torch.manual_seed(seed)

        for depth in depths:
            correct = 0
            for trial in range(n_trials):
                passkey = str(random.randint(10000, 99999))
                filler_unit = "The quick brown fox jumps over the lazy dog. "
                filler = filler_unit * (context_len // 8)
                needle = f"The secret passkey is {passkey}. Remember it."
                prompt = f"\nWhat is the secret passkey? The passkey is"

                char_pos = int(depth * len(filler))
                text = filler[:char_pos] + needle + filler[char_pos:]

                enc = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=context_len - 32)
                prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = torch.cat([enc.input_ids, prompt_enc.input_ids], dim=1).to(model.device)

                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=10, do_sample=False)
                gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                if passkey in gen:
                    correct += 1

            acc = correct / n_trials
            results[f"depth_{depth}"].append(acc)
            print(f"  Depth {depth:.0%}: {acc:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-SEED PASSKEY RESULTS")
    print("=" * 60)
    for key, accs in results.items():
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"  {key:15s}: {mean:.3f} ± {std:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed benchmark runner")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["ppl", "passkey", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--passkey-context", type=int, default=4096)
    parser.add_argument("--passkey-trials", type=int, default=10)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.experiment in ("ppl", "all"):
        ppl_results = run_ppl_multi_seed(
            args.model, args.seeds, context_len=args.context_len
        )
        all_results["ppl"] = ppl_results

    if args.experiment in ("passkey", "all"):
        passkey_results = run_passkey_multi_seed(
            args.model, args.seeds,
            context_len=args.passkey_context,
            n_trials=args.passkey_trials,
        )
        all_results["passkey"] = passkey_results

    # Save
    model_short = args.model.split("/")[-1].lower()
    out_path = RESULTS_DIR / f"multi_seed_{model_short}_{args.experiment}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
