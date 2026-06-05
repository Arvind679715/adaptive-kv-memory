"""Reproduce the Kaggle EXP15-EXP18 results from a clean environment.

Used to close paper open problem #7: third-party Kaggle artifacts ran
against an older AKV checkout. This script pins versions, recreates the
notebook execution order, and dumps a JSON summary that can be diffed
against the published numbers.

Why a script and not a notebook: notebooks hide the dependency graph
and re-run order. This file is the authoritative reproducible path.

Run on a Kaggle T4 / V100 / A100 instance::

    kaggle kernels init -p akv-repro
    cp scripts/repro_kaggle.py akv-repro/
    cd akv-repro && kaggle kernels push

Or locally with a fresh venv::

    python -m venv .venv-repro
    .venv-repro\\Scripts\\Activate.ps1   # or source on *nix
    pip install -r scripts/repro_kaggle_requirements.txt
    python scripts/repro_kaggle.py --out results/kaggle_repro.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

# Pinned versions used in the original EXP15-EXP18 Kaggle runs.
# Update only when re-running the experiments end-to-end.
PINNED = {
    "akv": "0.7.0",          # from the published Kaggle artifacts
    "torch": "2.4.0",
    "transformers": "4.45.2",
    "datasets": "3.0.1",
    "numpy": "1.26.4",
}

# Models used by EXP15-18 (subset \u2014 the full sweep takes several hours).
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# Datasets pulled by the original notebooks.
DATASETS = [
    ("THUDM/LongBench", "narrativeqa"),
    ("THUDM/LongBench", "qasper"),
    ("THUDM/LongBench", "multifieldqa_en"),
]


def _check_versions(strict: bool) -> dict[str, str]:
    """Verify installed package versions against the pinned set."""
    actual: dict[str, str] = {}
    mismatches: list[tuple[str, str, str]] = []
    for pkg, want in PINNED.items():
        try:
            mod = __import__(pkg)
            got = getattr(mod, "__version__", "?")
        except Exception as e:
            got = f"(import failed: {e})"
        actual[pkg] = got
        if got != want:
            mismatches.append((pkg, want, got))

    if mismatches:
        msg = "Version mismatch (-want +got):\n" + "\n".join(
            f"  {p}: -{w} +{g}" for p, w, g in mismatches
        )
        if strict:
            raise SystemExit(msg)
        print("WARNING:", msg, file=sys.stderr)
    return actual


def _run_one(model_id: str, dataset: tuple[str, str], cpu: bool) -> dict:
    """Run a single (model, dataset) cell and return summary metrics."""
    from akv import AKVCache
    from akv.evaluation import EvalConfig, PerplexityEvaluator

    print(f"=== {model_id} on {dataset} ===")
    t0 = time.perf_counter()

    cfg = EvalConfig(
        max_samples=10,            # smoke-level; bump for full repro
        max_length=2048,
        device="cpu" if cpu else "cuda",
    )
    evaluator = PerplexityEvaluator(model_id=model_id, config=cfg)
    cache = AKVCache(preset="medium")
    ppl = evaluator.evaluate_dataset(dataset_name=dataset[0], subset=dataset[1], cache=cache)
    dt = time.perf_counter() - t0
    print(f"    ppl={ppl:.3f}  ({dt:.1f}s)")
    return {"model": model_id, "dataset": list(dataset), "ppl": float(ppl), "wall_s": dt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/kaggle_repro.json"))
    ap.add_argument("--cpu", action="store_true", help="Force CPU (slow; for CI)")
    ap.add_argument("--strict-versions", action="store_true",
                    help="Fail if installed versions don't match PINNED")
    ap.add_argument("--smoke", action="store_true",
                    help="Only the first model+dataset combo (CI use)")
    args = ap.parse_args()

    summary: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "pinned_versions": PINNED,
        "installed_versions": _check_versions(args.strict_versions),
        "cells": [],
    }

    combos = list(zip(MODELS, DATASETS))
    if args.smoke:
        combos = combos[:1]

    for model, ds in combos:
        try:
            summary["cells"].append(_run_one(model, ds, cpu=args.cpu))
        except Exception as e:
            summary["cells"].append(
                {"model": model, "dataset": list(ds), "error": repr(e)}
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
