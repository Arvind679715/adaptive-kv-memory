"""End-to-end test: per-head bits from calibration reach the runtime."""
from __future__ import annotations

import pytest
import torch

from akv.drop_in import AKVCache, AKVLayer


def test_layer_accepts_per_head_bits():
    layer = AKVLayer(warm_bits=3, hot_budget=4, per_head_bits=[2, 3, 4, 2])
    assert layer.per_head_bits == [2, 3, 4, 2]


def test_layer_pool_built_lazily():
    layer = AKVLayer(warm_bits=3, hot_budget=4, per_head_bits=[2, 3, 4])
    # Drive a forward through `update` so the lazy quantizer init fires.
    k = torch.randn(1, 3, 8, 64)
    v = torch.randn(1, 3, 8, 64)
    out_k, out_v = layer.update(k, v)
    # After exceeding budget the per-bit pool must have one entry per distinct width.
    assert set(layer._quantizer_pool.keys()) == {2, 3, 4}
    # Output shape is preserved
    assert out_k.shape[-2] == out_v.shape[-2]


def test_per_head_quantize_uses_pool():
    """Force demotion and confirm the per-head path runs (not the global one)."""
    layer = AKVLayer(warm_bits=4, hot_budget=2, per_head_bits=[2, 3])
    k = torch.randn(1, 2, 4, 64)  # 4 tokens, budget 2 -> 2 must demote
    v = torch.randn(1, 2, 4, 64)
    layer.update(k, v)
    # 2 tokens should have been demoted to warm
    assert layer._warm_keys_fp16 is not None
    assert layer._warm_keys_fp16.shape[2] == 2


def test_per_head_bits_propagate_from_cache_to_layer():
    cache = AKVCache(warm_bits=3, hot_budget=4)
    cache._calibration_per_head_bits = {0: [2, 4]}
    k = torch.randn(1, 2, 6, 64)
    v = torch.randn(1, 2, 6, 64)
    cache.update(k, v, layer_idx=0)
    assert cache.layers[0].per_head_bits == [2, 4]


def test_per_head_bits_only_for_calibrated_layers():
    cache = AKVCache(warm_bits=3, hot_budget=4)
    cache._calibration_per_head_bits = {0: [2, 4]}  # only layer 0 calibrated
    k = torch.randn(1, 2, 6, 64)
    v = torch.randn(1, 2, 6, 64)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    assert cache.layers[0].per_head_bits == [2, 4]
    assert cache.layers[1].per_head_bits is None  # falls back to global warm_bits


def test_head_count_mismatch_falls_back_to_global():
    """Calibration ran with 4 heads but model has 2 -> use global, don't crash."""
    layer = AKVLayer(warm_bits=3, hot_budget=2, per_head_bits=[2, 3, 4, 2])
    k = torch.randn(1, 2, 4, 64)  # only 2 heads
    v = torch.randn(1, 2, 4, 64)
    # Should not raise even though per_head_bits has 4 entries
    layer.update(k, v)
    assert layer._warm_keys_fp16 is not None


def test_from_calibration_end_to_end_runtime():
    """JSON -> AKVCache -> forward pass actually exercises per-head bits."""
    import json
    import tempfile
    from pathlib import Path
    from akv.calibration import CalibrationReport

    report = CalibrationReport(
        model_name="x", model_type="llama",
        num_layers=4, num_kv_heads=2, head_dim=64,
        kv_outlier_ratio=3.0, attention_entropy_mean=2.0,
        attention_sink_strength=0.2,
        recommended_preset="balanced",
        recommended_protect_first=2,
        recommended_protect_last=4,
        recommended_average_bits=3.0,
        per_head_bits={0: [2, 3], 1: [4, 4]},
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        report.save(p)
        cache = AKVCache.from_calibration(p, hot_budget=2)

    # Run a forward through layers 0 and 1
    k = torch.randn(1, 2, 6, 64)
    v = torch.randn(1, 2, 6, 64)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    # Each layer carries its calibrated bits
    assert cache.layers[0].per_head_bits == [2, 3]
    assert cache.layers[1].per_head_bits == [4, 4]
    # Pools were built per distinct width
    assert set(cache.layers[0]._quantizer_pool.keys()) <= {2, 3}
    assert set(cache.layers[1]._quantizer_pool.keys()) <= {4}
