"""Tests for adaptive evictor."""
import pytest
import torch
from akv.evictor import AdaptiveEvictor, EvictionConfig
from akv.importance import ImportanceScorer, ImportanceConfig


class TestEvictionConfig:
    def test_defaults(self):
        cfg = EvictionConfig()
        assert cfg.max_seq_len_budget == 2048
        assert cfg.eviction_trigger_ratio == 0.9
        assert cfg.eviction_batch_size == 64


class TestShouldEvict:
    def test_below_budget(self):
        evictor = AdaptiveEvictor(EvictionConfig(max_seq_len_budget=100, eviction_trigger_ratio=0.9))
        assert not evictor.should_evict(80)

    def test_at_trigger(self):
        evictor = AdaptiveEvictor(EvictionConfig(max_seq_len_budget=100, eviction_trigger_ratio=0.9))
        assert evictor.should_evict(90)

    def test_above_trigger(self):
        evictor = AdaptiveEvictor(EvictionConfig(max_seq_len_budget=100, eviction_trigger_ratio=0.9))
        assert evictor.should_evict(95)


class TestComputeEviction:
    @pytest.fixture
    def evictor_with_scorer(self):
        scorer = ImportanceScorer(ImportanceConfig(
            initial_tokens_protected=2,
            recent_tokens_protected=2,
        ))
        attn = torch.zeros(1, 1, 1, 50)
        attn[0, 0, 0, 10] = 100.0
        attn[0, 0, 0, 20] = 80.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        scorer.update(attn, layer_idx=0)

        evictor = AdaptiveEvictor(
            EvictionConfig(max_seq_len_budget=30, eviction_batch_size=10, min_seq_len=4),
            scorer=scorer,
        )
        return evictor

    def test_no_eviction_needed(self, evictor_with_scorer):
        result = evictor_with_scorer.compute_eviction(0, current_seq_len=15)
        assert not result.should_evict
        assert result.num_evicted == 0
        assert result.keep_indices.numel() == 15

    def test_eviction_triggered(self, evictor_with_scorer):
        result = evictor_with_scorer.compute_eviction(0, current_seq_len=50, target_seq_len=30)
        assert result.should_evict
        assert result.num_evicted == 20
        assert result.new_seq_len == 30
        # Important tokens should be kept
        keep_list = result.keep_indices.tolist()
        assert 10 in keep_list  # high attention
        assert 20 in keep_list  # high attention
        assert 0 in keep_list   # protected initial
        assert 49 in keep_list  # protected recent

    def test_evict_keep_complement(self, evictor_with_scorer):
        result = evictor_with_scorer.compute_eviction(0, current_seq_len=50, target_seq_len=30)
        all_indices = set(result.keep_indices.tolist()) | set(result.evict_indices.tolist())
        assert all_indices == set(range(50))


class TestApplyEviction:
    def test_apply(self):
        evictor = AdaptiveEvictor()
        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)
        keep = torch.tensor([0, 2, 5, 7, 9])
        new_k, new_v = evictor.apply_eviction(keys, values, keep)
        assert new_k.shape == (1, 4, 5, 64)
        assert new_v.shape == (1, 4, 5, 64)
        # Check values are correct
        assert torch.allclose(new_k[:, :, 0, :], keys[:, :, 0, :])
        assert torch.allclose(new_k[:, :, 1, :], keys[:, :, 2, :])


class TestFallbackEviction:
    def test_eviction_without_scorer(self):
        evictor = AdaptiveEvictor(EvictionConfig(max_seq_len_budget=30, min_seq_len=4))
        result = evictor.compute_eviction(0, current_seq_len=50, target_seq_len=30)
        assert result.should_evict
        assert result.num_evicted == 20

    def test_stats_tracking(self):
        evictor = AdaptiveEvictor(EvictionConfig(max_seq_len_budget=30, min_seq_len=4))
        evictor.compute_eviction(0, current_seq_len=50, target_seq_len=30)
        assert evictor.stats["eviction_count"] == 1
        assert evictor.stats["total_evicted"] == 20
        evictor.reset_stats()
        assert evictor.stats["eviction_count"] == 0
