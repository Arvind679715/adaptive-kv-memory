"""Repro script for >=7B-parameter model evaluations (paper open problem #8).

Reviewers asked: "your perplexity numbers are all on <=3B models, do they
hold at 7B/8B/70B?" This script encodes the canonical evaluation matrix
so anyone with GPU budget can answer.

Hardware requirements (memory):
    - 7-8B at fp16:   ~16 GB  (1x A10/A6000/RTX 4090)
    - 13B at fp16:    ~28 GB  (1x A100-40GB)
    - 70B at fp16:    ~140 GB (4x A100-40GB or 2x H100-80GB)

Run::

    python scripts/repro_large_models.py --size 7b --out results/large_7b.json
    python scripts/repro_large_models.py --size 13b --out results/large_13b.json
    python scripts/repro_large_models.py --size 70b --quant int8 --out results/large_70b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Curated model matrix. One per family per size tier; gated weights need
# HF_TOKEN exported. Smaller alternatives are listed for tooling smoke
# tests; the actual paper numbers should use the primary IDs.
MODEL_MATRIX: dict[str, dict[str, list[str]]] = {
    "7b": {
        "primary": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "open": [  # ungated alternatives
            "Qwen/Qwen2.5-7B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
        ],
    },
    "13b": {
        "primary": [
            "meta-llama/Llama-2-13b-chat-hf",
            "Qwen/Qwen2.5-14B-Instruct",
        ],
        "open": [
            "Qwen/Qwen2.5-14B-Instruct",
        ],
    },
    "70b": {
        "primary": [
            "meta-llama/Llama-3.1-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "open": [
            "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
}

PRESETS = ["balanced", "memory", "quality"]
CONTEXT_LENGTHS = [4096, 16384, 32768]


def _evaluate(model_id: str, preset: str, ctx: int, quant: str | None) -> dict:
    """Run a single perplexity + memory evaluation cell."""
    from akv import AKVCache
    from akv.evaluation import EvalConfig, PerplexityEvaluator

    print(f"-- {model_id} preset={preset} ctx={ctx} quant={quant}")
    t0 = time.perf_counter()

    cfg = EvalConfig(
        max_samples=20,
        max_length=ctx,
        device="cuda",
        load_in_8bit=(quant == "int8"),
        load_in_4bit=(quant == "int4"),
    )
    evaluator = PerplexityEvaluator(model_id=model_id, config=cfg)
    cache = AKVCache(preset=preset)
    ppl = evaluator.evaluate(cache=cache)
    mem = cache.memory_usage() if hasattr(cache, "memory_usage") else {}
    dt = time.perf_counter() - t0

    return {
        "model": model_id,
        "preset": preset,
        "ctx": ctx,
        "quant": quant,
        "ppl": float(ppl),
        "memory": mem,
        "wall_s": dt,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=list(MODEL_MATRIX), default="7b")
    ap.add_argument("--gated", action="store_true",
                    help="Include gated weights (requires HF_TOKEN)")
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none",
                    help="Load the base model itself in lower precision")
    ap.add_argument("--out", type=Path, default=Path("results/large_models.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="Single model, single ctx, single preset")
    args = ap.parse_args()

    pool = MODEL_MATRIX[args.size]["primary" if args.gated else "open"]
    if args.smoke:
        pool = pool[:1]
        ctxs = [4096]
        presets = ["balanced"]
    else:
        ctxs = CONTEXT_LENGTHS
        presets = PRESETS

    cells: list[dict] = []
    for model in pool:
        for preset in presets:
            for ctx in ctxs:
                try:
                    cells.append(_evaluate(
                        model, preset, ctx,
                        quant=None if args.quant == "none" else args.quant,
                    ))
                except Exception as e:
                    cells.append({
                        "model": model, "preset": preset, "ctx": ctx,
                        "error": repr(e),
                    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "size_tier": args.size,
        "gated": args.gated,
        "base_quant": args.quant,
        "cells": cells,
    }, indent=2))
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
