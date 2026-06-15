"""Tests for importance scorer."""
import pytest
import torch
from akv.importance import ImportanceScorer, ImportanceConfig, ScoringStrategy


class TestImportanceConfig:
    def test_defaults(self):
        cfg = ImportanceConfig()
        assert cfg.strategy == ScoringStrategy.HYBRID
        assert cfg.decay_factor == 0.95
        assert cfg.initial_tokens_protected == 4
        assert cfg.recent_tokens_protected == 32


class TestImportanceScorer:
    @pytest.fixture
    def scorer(self):
        return ImportanceScorer(ImportanceConfig(
            initial_tokens_protected=2,
            recent_tokens_protected=4,
        ))

    @pytest.fixture
    def attention_weights(self):
        # (batch=1, heads=4, query=1, kv=16)
        torch.manual_seed(42)
        attn = torch.rand(1, 4, 1, 16)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn

    def test_update_creates_scores(self, scorer, attention_weights):
        scorer.update(attention_weights, layer_idx=0)
        scores = scorer.get_scores(0)
        assert scores is not None
        assert scores.shape[0] == 16

    def test_multiple_updates_accumulate(self, scorer, attention_weights):
        scorer.update(attention_weights, layer_idx=0)
        scores_1 = scorer.get_scores(0).clone()
        scorer.update(attention_weights, layer_idx=0)
        scores_2 = scorer.get_scores(0)
        # Scores should change after second update
        assert not torch.allclose(scores_1, scores_2)

    def test_step_count(self, scorer, attention_weights):
        assert scorer.step_count == 0
        scorer.update(attention_weights, layer_idx=0)
        assert scorer.step_count == 1
        scorer.update(attention_weights, layer_idx=1)
        assert scorer.step_count == 2

    def test_aggregated_scores(self, scorer, attention_weights):
        scorer.update(attention_weights, layer_idx=0)
        scorer.update(attention_weights, layer_idx=1)
        agg = scorer.get_aggregated_scores()
        assert agg is not None
        assert agg.shape[0] == 16

    def test_none_attention_is_noop(self, scorer):
        scorer.update(None, layer_idx=0)
        assert scorer.get_scores(0) is None

    def test_reset(self, scorer, attention_weights):
        scorer.update(attention_weights, layer_idx=0)
        scorer.reset()
        assert scorer.get_scores(0) is None
        assert scorer.step_count == 0


class TestTierAssignment:
    @pytest.fixture
    def scorer(self):
        s = ImportanceScorer(ImportanceConfig(
            initial_tokens_protected=2,
            recent_tokens_protected=2,
        ))
        # Create attention with known pattern: positions 5,6 get most attention
        attn = torch.zeros(1, 4, 1, 20)
        attn[:, :, :, 5] = 10.0
        attn[:, :, :, 6] = 8.0
        attn[:, :, :, 10] = 5.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        s.update(attn, layer_idx=0)
        return s

    def test_tier_sizes(self, scorer):
        hot, warm, cold = scorer.get_tier_assignments(0, seq_len=20, hot_budget=8, warm_budget=6)
        assert hot.numel() == 8
        assert warm.numel() == 6
        assert cold.numel() == 6
        # Total should equal seq_len
        assert hot.numel() + warm.numel() + cold.numel() == 20

    def test_protected_tokens_are_hot(self, scorer):
        hot, _, _ = scorer.get_tier_assignments(0, seq_len=20, hot_budget=8, warm_budget=6)
        hot_list = hot.tolist()
        # Initial tokens (0, 1) and recent tokens (18, 19) should be in hot
        assert 0 in hot_list
        assert 1 in hot_list
        assert 18 in hot_list
        assert 19 in hot_list

    def test_important_tokens_are_hot(self, scorer):
        hot, _, _ = scorer.get_tier_assignments(0, seq_len=20, hot_budget=8, warm_budget=6)
        hot_list = hot.tolist()
        # Positions 5, 6 had highest attention — should be in hot
        assert 5 in hot_list
        assert 6 in hot_list

    def test_indices_are_sorted(self, scorer):
        hot, warm, cold = scorer.get_tier_assignments(0, seq_len=20, hot_budget=8, warm_budget=6)
        assert (hot[1:] >= hot[:-1]).all()
        if warm.numel() > 1:
            assert (warm[1:] >= warm[:-1]).all()
        if cold.numel() > 1:
            assert (cold[1:] >= cold[:-1]).all()


class TestEvictionCandidates:
    def test_evicts_least_important(self):
        scorer = ImportanceScorer(ImportanceConfig(
            initial_tokens_protected=1,
            recent_tokens_protected=1,
        ))
        attn = torch.zeros(1, 1, 1, 10)
        attn[0, 0, 0, 5] = 100.0  # position 5 is very important
        attn = attn / attn.sum(dim=-1, keepdim=True)
        scorer.update(attn, layer_idx=0)

        candidates = scorer.get_eviction_candidates(0, seq_len=10, num_to_evict=3)
        assert candidates.numel() == 3
        # Position 5 should NOT be evicted, nor 0 (initial) or 9 (recent)
        evict_list = candidates.tolist()
        assert 5 not in evict_list
        assert 0 not in evict_list
        assert 9 not in evict_list

    def test_zero_eviction(self):
        scorer = ImportanceScorer()
        candidates = scorer.get_eviction_candidates(0, seq_len=10, num_to_evict=0)
        assert candidates.numel() == 0


class TestScoringStrategies:
    def test_all_strategies_produce_scores(self):
        attn = torch.rand(1, 4, 1, 32)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        for strategy in ScoringStrategy:
            scorer = ImportanceScorer(ImportanceConfig(strategy=strategy))
            scorer.update(attn, layer_idx=0)
            scores = scorer.get_scores(0)
            assert scores is not None
            assert scores.shape[0] == 32


class TestObservationWindowStrategy:
    """The observation-window strategy scores keys using only the most recent
    query positions (plus spatial pooling), which is more predictive of future
    decoding than an equal-weighted average over the whole prompt."""

    def _attn_with_shifting_focus(self, kv_len=20, q_len=8):
        # Early queries attend to key 3; the recent observation window (last 2
        # queries) attends to key 14. An equal-weight sum over all queries
        # dilutes key 14; an observation window surfaces it.
        attn = torch.full((1, 2, q_len, kv_len), 0.001)
        attn[:, :, :q_len - 2, 3] = 1.0
        attn[:, :, q_len - 2:, 14] = 1.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn

    def test_observation_window_surfaces_recent_focus(self):
        attn = self._attn_with_shifting_focus()

        obs = ImportanceScorer(ImportanceConfig(
            strategy=ScoringStrategy.OBSERVATION_WINDOW,
            observation_window=2,
            pooling_kernel=1,
        ))
        obs.update(attn, layer_idx=0)
        # Key 14 (recent focus) should outrank key 3 (stale focus).
        obs_scores = obs.get_scores(0)
        assert obs_scores[14] > obs_scores[3]

        # A plain equal-weight accumulation does the opposite (key 3 dominates).
        acc = ImportanceScorer(ImportanceConfig(
            strategy=ScoringStrategy.ATTENTION_ACCUMULATION,
        ))
        acc.update(attn, layer_idx=0)
        acc_scores = acc.get_scores(0)
        assert acc_scores[3] > acc_scores[14]

    def test_pooling_elevates_neighbours_of_a_spike(self):
        # Single-query attention spiking on key 10; pooling should raise its
        # immediate neighbours above unrelated distant keys.
        attn = torch.full((1, 1, 1, 20), 0.001)
        attn[:, :, :, 10] = 1.0
        attn = attn / attn.sum(dim=-1, keepdim=True)

        scorer = ImportanceScorer(ImportanceConfig(
            strategy=ScoringStrategy.OBSERVATION_WINDOW,
            observation_window=1,
            pooling_kernel=5,
        ))
        scorer.update(attn, layer_idx=0)
        scores = scorer.get_scores(0)
        assert scores[9] > scores[2]
        assert scores[11] > scores[2]
