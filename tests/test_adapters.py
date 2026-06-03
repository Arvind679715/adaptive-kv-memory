"""Tests for the model adapter registry."""
from __future__ import annotations

import pytest

from akv.adapters import (
    AdapterSpec, get_adapter, list_adapters, register_adapter, resolve_for_model,
)


def test_registry_has_core_families():
    families = {s.model_type for s in list_adapters()}
    for required in ["llama", "mistral", "qwen2", "gemma", "gemma2", "phi3", "mixtral"]:
        assert required in families, f"missing adapter for {required}"


def test_get_adapter_known():
    spec = get_adapter("llama")
    assert spec is not None
    assert spec.supported is True
    assert spec.default_preset in {"quality", "balanced", "compact"}


def test_get_adapter_unknown_returns_none():
    assert get_adapter("definitely_not_a_real_model_type") is None


def test_register_then_lookup():
    spec = AdapterSpec(
        model_type="acme_test_arch", family="ACME Test",
        supported=True, default_preset="quality",
    )
    register_adapter(spec)
    assert get_adapter("acme_test_arch") is spec


def test_mistral_has_sliding_window():
    spec = get_adapter("mistral")
    assert spec.sliding_window == 4096


def test_mla_marked_unsupported():
    spec = get_adapter("deepseek_v2")
    assert spec.kv_compressed is True
    assert spec.supported is False


def test_resolve_for_unknown_model_falls_back():
    class FakeConfig:
        model_type = "totally_unknown_arch_xyz"

    class FakeModel:
        config = FakeConfig()

    spec = resolve_for_model(FakeModel())
    assert spec.supported is True  # permissive fallback
    assert "totally_unknown_arch_xyz" in spec.model_type


def test_resolve_for_no_config():
    class FakeModel:
        pass

    spec = resolve_for_model(FakeModel())
    assert spec.model_type == "unknown"
    assert spec.supported is True


def test_describe_does_not_crash():
    for spec in list_adapters():
        s = spec.describe()
        assert spec.family in s
