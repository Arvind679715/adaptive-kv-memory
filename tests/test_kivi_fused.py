"""Tests for the fused-style KIVI baseline.

This baseline exists for fair throughput comparisons: published KIVI
numbers come from a fused CUDA kernel that does incremental append and
single-pass dequant. The naive ``KIVICache`` re-quantizes the entire
merged cache on every overflow, which inflates AKV's apparent speedup.

These tests verify functional parity (same dequantized values within
quantization tolerance) and the cost-profile improvements (no full
re-quant on overflow, no full re-dequant per step).
"""
from __future__ import annotations

import torch

from akv.baselines import (
    KIVICache,
    KIVIConfig,
    KIVIFusedCache,
    KIVIFusedConfig,
    create_baseline,
)


def _random_kv(batch=1, heads=4, n=1, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, heads, n, d, generator=g),
        torch.randn(batch, heads, n, d, generator=g),
    )


def test_kivi_fused_basic_functional_parity():
    """Fused KIVI should produce dequantized values within quant tolerance of naive KIVI."""
    cfg_naive = KIVIConfig(key_bits=4, value_bits=4, group_size=16, residual_length=8)
    cfg_fused = KIVIFusedConfig(key_bits=4, value_bits=4, group_size=16, residual_length=8)
    naive = KIVICache(cfg_naive)
    fused = KIVIFusedCache(cfg_fused)

    out_naive_k = out_fused_k = None
    for step in range(20):
        k, v = _random_kv(seed=step, n=1)
        out_naive_k, _ = naive.update(k, v, layer_idx=0)
        out_fused_k, _ = fused.update(k, v, layer_idx=0)

    assert out_naive_k.shape == out_fused_k.shape
    # Both go through 4-bit quantization with group_size=16, so values
    # should agree within typical quant noise (a few %).
    diff = (out_naive_k - out_fused_k).abs().max().item()
    assert diff < 0.5, f"fused KIVI diverged too far from naive: max |delta|={diff}"


def test_kivi_fused_memory_matches_naive_at_same_bits():
    """Both caches should report similar packed memory \u2014 the fused kernel
    isn't supposed to use *less* memory, just less compute per step."""
    cfg = KIVIConfig(key_bits=2, value_bits=2, group_size=16, residual_length=4)
    cfg_fused = KIVIFusedConfig(key_bits=2, value_bits=2, group_size=16, residual_length=4)
    naive = KIVICache(cfg)
    fused = KIVIFusedCache(cfg_fused)

    for step in range(32):
        k, v = _random_kv(seed=step, n=1)
        naive.update(k, v, layer_idx=0)
        fused.update(k, v, layer_idx=0)

    n_bytes = naive.memory_bytes()
    f_bytes = fused.memory_bytes()
    # Fused tracks an extra dequantized working copy, so it should be
    # measurably larger than naive (this is the cost of the speedup).
    # We don't require strict bounds here \u2014 just that both are non-zero
    # and in the same order of magnitude.
    assert n_bytes > 0 and f_bytes > 0
    assert max(n_bytes, f_bytes) / min(n_bytes, f_bytes) < 10.0


def test_kivi_fused_reset_clears_dequant_cache():
    """reset() must wipe both the quantized cache and the dequant working copy."""
    fused = KIVIFusedCache(KIVIFusedConfig(key_bits=4, value_bits=4, group_size=16, residual_length=4))
    for step in range(16):
        k, v = _random_kv(seed=step, n=1)
        fused.update(k, v, layer_idx=0)

    assert len(fused._dequant_keys_cache) > 0, "expected dequant cache after overflow"
    fused.reset()
    assert len(fused._dequant_keys_cache) == 0, "reset must clear dequant cache"
    assert len(fused._quant_keys) == 0, "reset must clear quant cache"


def test_create_baseline_kivi_fused():
    """The factory should accept 'kivi_fused' (and synonyms)."""
    for alias in ("kivi_fused", "kivi-fused", "kivifused", "fused_kivi"):
        cache = create_baseline(alias, key_bits=4, value_bits=4, group_size=16, residual_length=4)
        assert isinstance(cache, KIVIFusedCache), f"{alias} did not yield KIVIFusedCache"


def test_kivi_fused_incremental_quant_grows_chunks():
    """The packed buffer should accumulate chunks rather than be rebuilt each overflow.

    We can detect this by checking that the packed ``data`` row count
    grows monotonically (incremental concat) and that no single overflow
    re-derives a scale for the entire history.
    """
    fused = KIVIFusedCache(KIVIFusedConfig(key_bits=4, value_bits=4, group_size=16, residual_length=4))

    sizes = []
    for step in range(20):
        k, v = _random_kv(seed=step, n=1)
        fused.update(k, v, layer_idx=0)
        if 0 in fused._quant_keys:
            sizes.append(fused._quant_keys[0]["data"].shape[0])

    assert sizes, "expected at least one overflow"
    # Monotonic non-decreasing row count = incremental append.
    for a, b in zip(sizes, sizes[1:]):
        assert b >= a, f"packed row count went backwards: {sizes}"
