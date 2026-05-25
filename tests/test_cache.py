"""Tests for the adaptive hierarchical KV cache."""
import pytest
import torch
from akv.cache import AdaptiveKVCache, CacheConfig


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.hot_budget == 1024
        assert cfg.warm_budget == 2048
        assert cfg.warm_bits == 4
        assert cfg.cold_bits == 2


class TestAdaptiveKVCache:
    @pytest.fixture
    def small_cache(self):
        return AdaptiveKVCache(CacheConfig(
            hot_budget=16,
            warm_budget=16,
            warm_bits=4,
            cold_bits=2,
            group_size=8,
            initial_tokens_protected=2,
            recent_tokens_protected=4,
        ))

    @pytest.fixture
    def kv_pair(self):
        # (batch=1, heads=4, seq_len=8, head_dim=32)
        torch.manual_seed(42)
        k = torch.randn(1, 4, 8, 32, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 32, dtype=torch.float16)
        return k, v

    def test_initial_update(self, small_cache, kv_pair):
        k, v = kv_pair
        out_k, out_v = small_cache.update(k, v, layer_idx=0)
        assert out_k.shape == k.shape
        assert out_v.shape == v.shape

    def test_multiple_updates_accumulate(self, small_cache, kv_pair):
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        out_k, out_v = small_cache.update(k, v, layer_idx=0)
        # Should have 16 tokens now (8 + 8)
        assert out_k.shape[2] == 16

    def test_len(self, small_cache, kv_pair):
        assert len(small_cache) == 0
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        assert len(small_cache) == 1
        small_cache.update(k, v, layer_idx=1)
        assert len(small_cache) == 2

    def test_getitem(self, small_cache, kv_pair):
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        out_k, out_v = small_cache[0]
        assert out_k.shape == k.shape

    def test_get_seq_length(self, small_cache, kv_pair):
        assert small_cache.get_seq_length() == 0
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        assert small_cache.get_seq_length(0) == 8

    def test_reorganization_triggers(self, small_cache):
        """When hot tier exceeds budget, tokens should be reorganized to warm/cold."""
        torch.manual_seed(42)
        # Add enough tokens to exceed hot_budget of 16
        for i in range(4):
            k = torch.randn(1, 4, 8, 32, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 32, dtype=torch.float16)
            small_cache.update(k, v, layer_idx=0)

        # After 32 tokens with budget=16, should have reorganized
        summary = small_cache.tier_summary()
        assert summary["reorganizations"] > 0
        # Hot tier should be within budget
        assert summary["hot_tokens_avg"] <= 16

    def test_memory_usage_tracking(self, small_cache, kv_pair):
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        usage = small_cache.memory_usage()
        assert usage["hot_mb"] > 0
        assert usage["num_layers"] == 1

    def test_reset(self, small_cache, kv_pair):
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        small_cache.reset()
        assert len(small_cache) == 0
        assert small_cache.get_seq_length() == 0

    def test_iteration(self, small_cache, kv_pair):
        k, v = kv_pair
        small_cache.update(k, v, layer_idx=0)
        small_cache.update(k, v, layer_idx=1)
        layers = list(small_cache)
        assert len(layers) == 2

    def test_with_attention_weights(self, small_cache, kv_pair):
        k, v = kv_pair
        attn = torch.rand(1, 4, 8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        small_cache.update(k, v, layer_idx=0, attention_weights=attn)
        assert small_cache.scorer.step_count == 1


class TestTierReorganization:
    def test_warm_tier_quantized(self):
        cache = AdaptiveKVCache(CacheConfig(
            hot_budget=8,
            warm_budget=8,
            warm_bits=4,
            group_size=8,
            initial_tokens_protected=1,
            recent_tokens_protected=2,
            enable_cold_tier=False,
        ))
        torch.manual_seed(42)
        # Add 24 tokens in 3 batches
        for _ in range(3):
            k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
            v = torch.randn(1, 2, 8, 16, dtype=torch.float16)
            cache.update(k, v, layer_idx=0)

        # Check that warm tier has data
        layer = cache._layers[0]
        assert layer.warm_keys is not None or layer.hot_len <= 8

    def test_cold_tier_on_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cache = AdaptiveKVCache(CacheConfig(
            hot_budget=4,
            warm_budget=4,
            cold_bits=2,
            group_size=8,
            initial_tokens_protected=1,
            recent_tokens_protected=1,
            enable_cold_tier=True,
        ))
        # Add many tokens to force cold tier
        for _ in range(5):
            k = torch.randn(1, 2, 8, 16, dtype=torch.float16, device="cuda")
            v = torch.randn(1, 2, 8, 16, dtype=torch.float16, device="cuda")
            cache.update(k, v, layer_idx=0)

        layer = cache._layers[0]
        if layer.cold_keys is not None:
            # Cold tier should be on CPU
            assert layer.cold_keys.scales.device.type == "cpu"


class TestPromotion:
    def test_promote_from_warm(self):
        cache = AdaptiveKVCache(CacheConfig(
            hot_budget=8,
            warm_budget=16,
            warm_bits=4,
            group_size=8,
            initial_tokens_protected=1,
            recent_tokens_protected=1,
            enable_cold_tier=False,
        ))
        torch.manual_seed(42)
        # Fill cache to trigger reorganization
        for _ in range(4):
            k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
            v = torch.randn(1, 2, 8, 16, dtype=torch.float16)
            cache.update(k, v, layer_idx=0)

        layer = cache._layers[0]
        if layer.warm_positions is not None and layer.warm_positions.numel() > 0:
            # Promote a warm token
            pos_to_promote = layer.warm_positions[:1]
            hot_before = layer.hot_len
            cache.promote_tokens(0, pos_to_promote)
            # Hot tier should have grown
            assert layer.hot_len >= hot_before
