"""Tests for baseline KV cache implementations."""
import pytest
import torch
from akv.baselines import (
    FullCache, H2OCache, H2OConfig,
    KIVICache, KIVIConfig,
    SnapKVCache, SnapKVConfig,
    ScissorHandsCache, ScissorHandsConfig,
    create_baseline,
)


@pytest.fixture
def kv_pair():
    torch.manual_seed(42)
    k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    return k, v


@pytest.fixture
def attn_weights():
    torch.manual_seed(42)
    a = torch.rand(1, 4, 8, 8)
    return a / a.sum(dim=-1, keepdim=True)


class TestFullCache:
    def test_update(self, kv_pair):
        cache = FullCache()
        k, v = kv_pair
        out_k, out_v = cache.update(k, v, 0)
        assert out_k.shape == k.shape
        assert out_v.shape == v.shape

    def test_accumulate(self, kv_pair):
        cache = FullCache()
        k, v = kv_pair
        cache.update(k, v, 0)
        out_k, _ = cache.update(k, v, 0)
        assert out_k.shape[2] == 16

    def test_memory(self, kv_pair):
        cache = FullCache()
        k, v = kv_pair
        cache.update(k, v, 0)
        assert cache.memory_bytes() > 0

    def test_reset(self, kv_pair):
        cache = FullCache()
        k, v = kv_pair
        cache.update(k, v, 0)
        cache.reset()
        assert cache.get_seq_length() == 0


class TestH2OCache:
    def test_within_budget(self, kv_pair, attn_weights):
        cache = H2OCache(H2OConfig(budget=100))
        k, v = kv_pair
        out_k, _ = cache.update(k, v, 0, attn_weights)
        assert out_k.shape[2] == 8

    def test_eviction(self, attn_weights):
        cache = H2OCache(H2OConfig(budget=16, heavy_hitter_k=8, recent_window=8))
        torch.manual_seed(42)
        for _ in range(4):
            k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            a = torch.rand(1, 4, 8, cache.get_seq_length(0) + 8)
            a = a / a.sum(dim=-1, keepdim=True)
            cache.update(k, v, 0, a)
        assert cache.get_seq_length(0) <= 16

    def test_memory_bounded(self):
        cache = H2OCache(H2OConfig(budget=16))
        torch.manual_seed(42)
        for _ in range(10):
            k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            a = torch.rand(1, 4, 8, cache.get_seq_length(0) + 8)
            a = a / a.sum(dim=-1, keepdim=True)
            cache.update(k, v, 0, a)
        assert cache.get_seq_length(0) <= 16


class TestKIVICache:
    def test_basic(self, kv_pair):
        cache = KIVICache(KIVIConfig(residual_length=16))
        k, v = kv_pair
        out_k, out_v = cache.update(k, v, 0)
        assert out_k.shape == k.shape

    def test_quantize_overflow(self):
        cache = KIVICache(KIVIConfig(key_bits=2, value_bits=2, residual_length=8, group_size=8))
        torch.manual_seed(42)
        for _ in range(4):
            k = torch.randn(1, 4, 8, 16, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 16, dtype=torch.float16)
            cache.update(k, v, 0)
        # Should have quantized some tokens
        assert cache.get_seq_length(0) == 32
        assert 0 in cache._quant_keys  # quantized portion exists

    def test_memory_compressed(self):
        cache_full = FullCache()
        cache_kivi = KIVICache(KIVIConfig(key_bits=2, value_bits=2, residual_length=16, group_size=16))
        torch.manual_seed(42)
        for _ in range(8):
            k = torch.randn(1, 4, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 16, 64, dtype=torch.float16)
            cache_full.update(k, v, 0)
            cache_kivi.update(k, v, 0)
        # KIVI should use less memory
        assert cache_kivi.memory_bytes() < cache_full.memory_bytes()


class TestSnapKVCache:
    def test_no_compression_under_budget(self, kv_pair, attn_weights):
        cache = SnapKVCache(SnapKVConfig(budget=100))
        k, v = kv_pair
        out_k, _ = cache.update(k, v, 0, attn_weights)
        assert out_k.shape[2] == 8

    def test_compression(self):
        cache = SnapKVCache(SnapKVConfig(budget=16, observation_window=8))
        torch.manual_seed(42)
        # Add many tokens — SnapKV compresses when seq > budget AND attention is provided
        all_k, all_v = [], []
        for _ in range(4):
            k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            all_k.append(k)
            all_v.append(v)
        # Feed all at once with attention (simulating prefill)
        big_k = torch.cat(all_k, dim=2)
        big_v = torch.cat(all_v, dim=2)
        a = torch.rand(1, 4, 32, 32)
        a = a / a.sum(dim=-1, keepdim=True)
        cache.update(big_k, big_v, 0, a)
        # Should have compressed
        assert cache.get_seq_length(0) <= 16


class TestScissorHandsCache:
    def test_basic(self, kv_pair, attn_weights):
        cache = ScissorHandsCache(ScissorHandsConfig(budget=100))
        k, v = kv_pair
        out_k, _ = cache.update(k, v, 0, attn_weights)
        assert out_k.shape[2] == 8

    def test_eviction_with_history(self):
        cache = ScissorHandsCache(ScissorHandsConfig(
            budget=16, history_window=4, recent_window=4,
        ))
        torch.manual_seed(42)
        for _ in range(6):
            k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            current_len = cache.get_seq_length(0) + 8
            a = torch.rand(1, 4, 8, current_len)
            a = a / a.sum(dim=-1, keepdim=True)
            cache.update(k, v, 0, a)
        assert cache.get_seq_length(0) <= 16


class TestCreateBaseline:
    def test_full(self):
        assert isinstance(create_baseline("full"), FullCache)

    def test_h2o(self):
        assert isinstance(create_baseline("h2o", budget=512), H2OCache)

    def test_kivi(self):
        assert isinstance(create_baseline("kivi", key_bits=4), KIVICache)

    def test_unknown(self):
        with pytest.raises(ValueError):
            create_baseline("unknown_method")
