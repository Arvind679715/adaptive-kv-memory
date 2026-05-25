"""Tests for HuggingFace integration."""
import pytest
import torch
from akv.cache import CacheConfig
from akv.integration import HFAdaptiveCache, _sample


class TestHFAdaptiveCache:
    @pytest.fixture
    def hf_cache(self):
        return HFAdaptiveCache(CacheConfig(
            hot_budget=32,
            warm_budget=32,
            warm_bits=4,
            group_size=8,
        ))

    def test_update(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        out_k, out_v = hf_cache.update(k, v, layer_idx=0)
        assert out_k.shape == k.shape

    def test_seen_tokens(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        hf_cache.update(k, v, layer_idx=0)
        assert hf_cache.seen_tokens == 8
        hf_cache.update(k, v, layer_idx=0)
        # seen_tokens incremented on every layer_idx=0 update
        assert hf_cache.seen_tokens == 16

    def test_seq_length(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        hf_cache.update(k, v, layer_idx=0)
        assert hf_cache.get_seq_length(0) == 8

    def test_len(self, hf_cache):
        assert len(hf_cache) == 0
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        hf_cache.update(k, v, layer_idx=0)
        assert len(hf_cache) == 1

    def test_reset(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        hf_cache.update(k, v, layer_idx=0)
        hf_cache.reset()
        assert len(hf_cache) == 0
        assert hf_cache.seen_tokens == 0

    def test_memory_usage(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        hf_cache.update(k, v, layer_idx=0)
        usage = hf_cache.memory_usage()
        assert usage["hot_mb"] > 0

    def test_with_cache_kwargs(self, hf_cache):
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        attn = torch.rand(1, 4, 8, 8)
        hf_cache.update(k, v, layer_idx=0, cache_kwargs={"attention_weights": attn})
        assert hf_cache.inner_cache.scorer.step_count == 1


class TestSampling:
    def test_greedy(self):
        logits = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
        token = _sample(logits, temperature=0, top_p=1.0)
        assert token.item() == 2

    def test_temperature(self):
        torch.manual_seed(42)
        logits = torch.tensor([[0.1, 0.2, 10.0, 0.3]])
        token = _sample(logits, temperature=0.01, top_p=1.0)
        # Very low temperature should be near-greedy
        assert token.item() == 2

    def test_top_p(self):
        torch.manual_seed(42)
        logits = torch.tensor([[0.0, 0.0, 100.0, 0.0]])
        token = _sample(logits, temperature=1.0, top_p=0.1)
        # With top_p=0.1 and one dominant logit, should pick it
        assert token.item() == 2
