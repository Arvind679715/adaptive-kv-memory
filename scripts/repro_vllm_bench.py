"""Repro script for vLLM continuous-batching benchmarks (paper open problem #9).

This is a smoke + scaling harness for AKV's vLLM integration. It does
not aim to reproduce vLLM's own numbers \u2014 we measure AKV-with-vLLM vs
vanilla-vLLM at matched batch sizes, so the *delta* is the headline
number.

Required: vLLM 0.6.x (older versions used a different attention API).
A GPU with >=16 GB is recommended for the 7B configs.

Run::

    pip install vllm==0.6.3.post1
    python scripts/repro_vllm_bench.py --smoke    # 1 model, 1 batch size
    python scripts/repro_vllm_bench.py --full     # full sweep (~30 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PINNED_VLLM = "0.6.3.post1"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # always-on smoke target
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
BATCH_SIZES = [1, 4, 16, 32, 64]
PROMPT_LENS = [512, 2048, 8192]
DECODE_LEN = 128


def _import_vllm():
    try:
        import vllm  # noqa: F401
        return True
    except Exception as e:
        print(f"vLLM not importable: {e}", file=sys.stderr)
        return False


def _run_one(model_id: str, batch: int, plen: int, use_akv: bool) -> dict:
    """Single config: returns throughput in tokens/s and p50/p99 latency ms."""
    from vllm import LLM, SamplingParams

    print(f"  model={model_id} batch={batch} plen={plen} akv={use_akv}")
    # The integration is exposed via env var so we can flip it without
    # touching vLLM internals from this script.
    import os
    os.environ["AKV_VLLM_ENABLE"] = "1" if use_akv else "0"

    llm = LLM(model=model_id, gpu_memory_utilization=0.8, max_model_len=plen + DECODE_LEN + 64)
    prompts = ["hello " * (plen // 6)] * batch
    sp = SamplingParams(max_tokens=DECODE_LEN, temperature=0.0)

    # Warmup
    llm.generate(prompts[: min(batch, 4)], sp)

    latencies: list[float] = []
    n_iters = 3
    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ti = time.perf_counter()
        outs = llm.generate(prompts, sp)
        latencies.append(time.perf_counter() - ti)
        total_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
    wall = time.perf_counter() - t0

    latencies.sort()
    return {
        "model": model_id,
        "batch": batch,
        "prompt_len": plen,
        "decode_len": DECODE_LEN,
        "use_akv": use_akv,
        "throughput_tok_per_s": total_tokens / wall,
        "p50_latency_ms": latencies[len(latencies) // 2] * 1000.0,
        "p99_latency_ms": latencies[-1] * 1000.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/vllm_bench.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="0.5B model, batch=4, plen=512 only")
    ap.add_argument("--full", action="store_true", help="Full sweep")
    args = ap.parse_args()

    summary: dict = {"pinned_vllm": PINNED_VLLM, "cells": []}
    if not _import_vllm():
        summary["error"] = "vllm not importable; install vllm==" + PINNED_VLLM
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        return 1

    if args.smoke or not args.full:
        configs = [(MODELS[0], 4, 512)]
    else:
        configs = [(m, b, p) for m in MODELS for b in BATCH_SIZES for p in PROMPT_LENS]

    for model, batch, plen in configs:
        for use_akv in (False, True):
            try:
                summary["cells"].append(_run_one(model, batch, plen, use_akv))
            except Exception as e:
                summary["cells"].append({
                    "model": model, "batch": batch, "prompt_len": plen,
                    "use_akv": use_akv, "error": repr(e),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
