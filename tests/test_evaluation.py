"""Tests for evaluation framework."""
import pytest
import torch
from akv.evaluation import (
    MethodConfig, AdaptiveKVCacheWrapper, get_standard_methods,
    measure_memory_scaling, run_ablation,
)
from akv.cache import CacheConfig
from akv.baselines import FullCache


class TestMethodConfig:
    def test_create_full(self):
        cfg = MethodConfig(name="full", method_type="full")
        cache = cfg.create_cache()
        assert isinstance(cache, FullCache)

    def test_create_akv(self):
        cfg = MethodConfig(name="akv", method_type="akv", params={
            "hot_budget": 64, "warm_budget": 64, "warm_bits": 4,
            "group_size": 32,
        })
        cache = cfg.create_cache()
        assert isinstance(cache, AdaptiveKVCacheWrapper)


class TestAKVWrapper:
    def test_interface(self):
        wrapper = AdaptiveKVCacheWrapper(CacheConfig(
            hot_budget=32, warm_budget=32, warm_bits=4, group_size=8,
        ))
        k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
        v = torch.randn(1, 2, 8, 16, dtype=torch.float16)
        out_k, out_v = wrapper.update(k, v, 0)
        assert out_k.shape == k.shape
        assert wrapper.get_seq_length(0) == 8
        assert wrapper.memory_bytes() > 0
        wrapper.reset()
        assert wrapper.get_seq_length(0) == 0


class TestGetStandardMethods:
    def test_returns_methods(self):
        methods = get_standard_methods(budget=256)
        assert len(methods) >= 5
        names = [m.name for m in methods]
        assert any("Full" in n for n in names)
        assert any("H2O" in n for n in names)
        assert any("KIVI" in n for n in names)
        assert any("AKV" in n for n in names)


class TestMemoryScaling:
    def test_small_scale(self):
        methods = [
            MethodConfig(name="Full", method_type="full"),
            MethodConfig(name="H2O", method_type="h2o", params={"budget": 32}),
        ]
        results = measure_memory_scaling(
            methods=methods,
            seq_lens=[32, 64],
            num_layers=2,
            num_heads=2,
            head_dim=16,
        )
        assert len(results) == 4  # 2 methods * 2 seq_lens
        assert all("memory_mb" in r for r in results)
        assert all("compression_ratio" in r for r in results)


class TestAblation:
    def test_bits_ablation(self):
        results = run_ablation(
            "bits",
            seq_len=64,
            num_layers=2,
            num_heads=2,
            head_dim=16,
            base_config={
                "hot_budget": 32, "warm_budget": 32, "warm_bits": 4,
                "cold_bits": 2, "group_size": 16, "enable_cold_tier": False,
            },
        )
        assert len(results) == 3  # 2, 4, 8 bit
        names = [r["name"] for r in results]
        assert "warm_2bit" in names
        assert "warm_4bit" in names
        assert "warm_8bit" in names

    def test_budget_ablation(self):
        results = run_ablation(
            "budget",
            seq_len=64,
            num_layers=2,
            num_heads=2,
            head_dim=16,
            base_config={
                "hot_budget": 32, "warm_budget": 32, "warm_bits": 4,
                "cold_bits": 2, "group_size": 16, "enable_cold_tier": False,
            },
        )
        assert len(results) == 5  # 256, 512, 1024, 2048, 4096
