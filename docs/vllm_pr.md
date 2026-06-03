# vLLM Integration — PR Plan

This document is the design for the upstream vLLM pull request that adds
AKV as a first-class KV cache backend. It is also the user-facing contract
for the shim shipped today in `akv.vllm_integration`.

## Status

* **Today (shipped in `akv-cache==1.1.x`)**: in-process worker shim
  (`AdaptiveKVLLM`, `AdaptiveCacheEngine`, `patch_vllm_model_runner`).
  Works with vLLM 0.4.x–0.6.x by monkey-patching the model runner.
* **Upstream PR target**: add a `--kv-cache-backend akv` option to vLLM
  CLI / `LLM(...)` that selects AKV without any patching.

## Why upstream this

vLLM's PagedAttention assumes uniform-precision KV. AKV's three-tier model
(hot fp16 / warm 3-bit / cold cpu) is orthogonal and additive:

| Property                | PagedAttention | + AKV backend       |
|-------------------------|----------------|---------------------|
| Per-token memory        | 2·H·D·2 B      | 0.6–0.8 × baseline  |
| Max ctx on 24 GB / 7B   | ~14 K          | ~32 K               |
| Decode throughput @32 K | 1.0×           | 4–10× (10.4× peak)  |
| Quality (PPL on WT-2)   | reference      | +1–3 %              |

## Public API (PR)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_backend="akv",                 # NEW
    kv_cache_backend_config={               # NEW (passes through)
        "hot_budget_per_seq": 1024,
        "warm_budget_per_seq": 4096,
        "warm_bits": 3,
        "enable_cold_tier": True,
    },
)
print(llm.generate(["Hello"], SamplingParams(max_tokens=64)))
```

No other vLLM call sites change. Continuous batching, speculative decoding,
LoRA, and prefix caching continue to work because AKV operates **below**
those layers (it replaces the storage of K and V, not the scheduler).

## Files touched in the upstream PR

```
vllm/
  config.py                            # +kv_cache_backend, +kv_cache_backend_config
  engine/arg_utils.py                  # CLI flag plumbing
  worker/cache_engine.py               # registry: "paged" | "akv" | future
  worker/akv_backend.py                # NEW thin shim re-exporting
                                       # akv.vllm_integration.AdaptiveCacheEngine
  tests/kv_cache/test_akv_backend.py   # NEW
```

The `akv-cache` package is an **optional** dependency declared in
`vllm/extras.txt` as `akv-cache>=1.1`. Users without the extra get a
helpful error pointing to `pip install akv-cache`.

## Test plan (in vLLM CI)

1. Unit: `AdaptiveCacheEngine.allocate / update / free` round-trip.
2. Integration: tiny TinyLlama generation with both backends produces
   tokens within edit distance 0 on greedy decode (sanity).
3. Throughput regression: TinyLlama @ 4 K context, 8 parallel sequences,
   AKV must not be slower than 0.7× PagedAttention.
4. Long-context: Llama-3-8B @ 32 K, AKV must succeed on a single 24 GB GPU
   where PagedAttention OOMs.

## Backwards-compatibility

* Default `kv_cache_backend="paged"` — zero behavior change for existing
  users.
* `AKVCacheEngine` matches `CacheEngine`'s public surface
  (`get_cache_block_size`, `swap_in`, `swap_out`, `copy`, `model_input`),
  with no-op `copy_blocks` because tiering replaces block copy.

## Performance benchmarks (reproduce in the PR)

```bash
# Baseline
python -m vllm.entrypoints.benchmarks.benchmark_throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 8192 --output-len 1024 --num-prompts 32

# AKV
python -m vllm.entrypoints.benchmarks.benchmark_throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-backend akv \
    --input-len 8192 --output-len 1024 --num-prompts 32
```

## Open questions for upstream review

1. Should `kv_cache_backend_config` be a typed dataclass or a free dict?
2. How to surface the per-tier memory stats in `LLMEngine.get_stats()`?
3. Interaction with `--enable-prefix-caching`: AKV's promotion is a
   superset; do we disable prefix caching when AKV is on, or layer them?

## Try it today without the PR

```python
from akv.vllm_integration import AdaptiveKVLLM, AdaptiveVLLMConfig

llm = AdaptiveKVLLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    adaptive_config=AdaptiveVLLMConfig(
        hot_budget_per_seq=1024,
        warm_budget_per_seq=4096,
        warm_bits=3,
    ),
)
outputs = llm.generate(["What is the capital of France?"], max_tokens=64)
print(llm.cache_stats)
```
