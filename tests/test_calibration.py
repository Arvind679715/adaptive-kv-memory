"""Tests for the calibration pipeline (CPU only, tiny random model)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from akv.calibration import (
    CalibrationReport, HeadSensitivity, _measure_quant_error, calibrate_model,
)


def test_quant_error_decreases_with_bits():
    x = torch.randn(2, 4, 32, 128)
    e2 = _measure_quant_error(x, 2)
    e3 = _measure_quant_error(x, 3)
    e4 = _measure_quant_error(x, 4)
    # More bits -> lower error. Allow small floor for ties on synthetic data.
    assert e4 <= e3 + 1e-3
    assert e3 <= e2 + 1e-3


def test_head_sensitivity_best_bits():
    # Sensitive head: 2b is much worse than 4b, 3b is close.
    sensitive = HeadSensitivity(0, 0, err_2bit=0.5, err_3bit=0.06, err_4bit=0.05)
    assert sensitive.best_bits_for_budget(3.5) == 3
    # 2b unusable -> falls back to 3 (next within tolerance)
    assert sensitive.best_bits_for_budget(2.0) == 3
    # Robust head: 2b is fine.
    robust = HeadSensitivity(0, 0, err_2bit=0.06, err_3bit=0.05, err_4bit=0.05)
    assert robust.best_bits_for_budget(2.0) == 2


def test_report_json_roundtrip():
    r = CalibrationReport(
        model_name="x", model_type="llama",
        num_layers=4, num_kv_heads=2, head_dim=16,
        kv_outlier_ratio=3.0, attention_entropy_mean=2.0,
        attention_sink_strength=0.2,
        recommended_preset="balanced",
        recommended_protect_first=2,
        recommended_protect_last=16,
        recommended_average_bits=3.0,
        per_head_bits={0: [3, 4], 1: [2, 3]},
        sensitivities=[HeadSensitivity(0, 0, 0.1, 0.05, 0.02)],
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calib.json"
        r.save(p)
        loaded = CalibrationReport.load(p)
    assert loaded.model_name == r.model_name
    assert loaded.per_head_bits == r.per_head_bits
    assert loaded.sensitivities[0].err_3bit == 0.05


def test_calibrate_tiny_random_model():
    """End-to-end smoke test with a tiny random Llama (no network)."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = AutoModelForCausalLM.from_config(cfg).eval()
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception as e:
        pytest.skip(f"Could not load tiny test model: {e}")

    report = calibrate_model(
        model, tok,
        sample_texts=["Hello world. " * 20],
        max_length=64, max_layers_to_probe=2,
    )
    assert report.num_layers > 0
    assert report.recommended_preset in {"quality", "balanced", "compact"}
    assert 2.0 <= report.recommended_average_bits <= 4.0
    # JSON round-trip
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.json"
        report.save(p)
        assert json.loads(p.read_text())["model_type"] == report.model_type


def test_akvcache_from_calibration():
    """Verify the from_calibration classmethod consumes a report."""
    pytest.importorskip("transformers")
    from akv import AKVCache

    r = CalibrationReport(
        model_name="x", model_type="llama",
        num_layers=8, num_kv_heads=4, head_dim=64,
        kv_outlier_ratio=3.0, attention_entropy_mean=2.0,
        attention_sink_strength=0.2,
        recommended_preset="balanced",
        recommended_protect_first=2,
        recommended_protect_last=16,
        recommended_average_bits=3.0,
        per_head_bits={0: [3, 4, 3, 3]},
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        r.save(p)
        cache = AKVCache.from_calibration(p)
    assert cache is not None
    # Our overrides should have been applied
    assert getattr(cache, "_calibration_per_head_bits", None) is not None
