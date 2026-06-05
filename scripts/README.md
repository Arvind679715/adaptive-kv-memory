# Reproduction Scripts

These three scripts close the three "open" paper problems that depend on
infrastructure we don't run in CI. Each is self-contained and writes a
JSON summary to `results/` that can be diffed against published numbers.

| Script | Closes paper open problem | Hardware | Wall time |
|--------|---------------------------|----------|-----------|
| `repro_kaggle.py` | #7 — Kaggle EXP15-EXP18 re-run | T4 / V100 | ~30 min |
| `repro_large_models.py` | #8 — perplexity at >=7B params | A100-40GB (7B), 4xA100 (70B) | ~1-6 h |
| `repro_vllm_bench.py` | #9 — vLLM continuous batching | RTX 4090 / A100 | ~30 min |

## Quick smoke (any GPU, ~1 min each)

```powershell
python scripts/repro_kaggle.py --smoke --cpu --out results/smoke_kaggle.json
python scripts/repro_large_models.py --smoke --out results/smoke_large.json
python scripts/repro_vllm_bench.py --smoke --out results/smoke_vllm.json
```

## Full repro

Each script accepts `--full` (or its non-`--smoke` equivalent) to run the
canonical matrix used in the paper. See per-script `--help` for size
tiers, model gating, and quantization flags.

## Pinned versions

- `repro_kaggle.py` pins via `repro_kaggle_requirements.txt` (akv-cache
  0.7.0 + transformers 4.45.2). Use `--strict-versions` to fail on drift.
- `repro_large_models.py` runs against whatever is installed; the model
  IDs are the contract.
- `repro_vllm_bench.py` pins vLLM 0.6.3.post1; older versions used a
  different attention API and the AKV adapter won't load.
