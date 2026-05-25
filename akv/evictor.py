"""Adaptive eviction policies for KV cache management.

Decides when and which tokens to evict from the cache based on
importance scores, budget constraints, and memory pressure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from akv.importance import ImportanceScorer

logger = logging.getLogger(__name__)


class EvictionPolicy(str, Enum):
    IMPORTANCE = "importance"
    WINDOW_IMPORTANCE = "window_importance"
    BUDGET_AWARE = "budget_aware"


@dataclass
class EvictionConfig:
    policy: EvictionPolicy = EvictionPolicy.BUDGET_AWARE
    max_seq_len_budget: int = 2048
    eviction_trigger_ratio: float = 0.9
    eviction_batch_size: int = 64
    min_seq_len: int = 128


class EvictionResult:
    """Result of an eviction decision."""
    __slots__ = ('should_evict', 'evict_indices', 'keep_indices',
                 'num_evicted', 'new_seq_len')

    def __init__(
        self,
        should_evict: bool,
        evict_indices: torch.Tensor,
        keep_indices: torch.Tensor,
        num_evicted: int,
        new_seq_len: int,
    ):
        self.should_evict = should_evict
        self.evict_indices = evict_indices
        self.keep_indices = keep_indices
        self.num_evicted = num_evicted
        self.new_seq_len = new_seq_len


class AdaptiveEvictor:
    """Decides when and what to evict from the KV cache.

    Works with ImportanceScorer to evict least-important tokens when
    the cache exceeds its budget. Supports different policies:

    - importance: purely importance-based eviction
    - window_importance: keep a sliding window + top-k important tokens
    - budget_aware: evict when memory budget is exceeded, with batched eviction
    """

    def __init__(
        self,
        config: Optional[EvictionConfig] = None,
        scorer: Optional[ImportanceScorer] = None,
    ):
        self.config = config or EvictionConfig()
        self.scorer = scorer
        self._eviction_count = 0
        self._total_evicted = 0

    def should_evict(self, current_seq_len: int) -> bool:
        """Check if eviction should be triggered based on current seq length."""
        cfg = self.config
        trigger_len = int(cfg.max_seq_len_budget * cfg.eviction_trigger_ratio)
        return current_seq_len >= trigger_len

    def compute_eviction(
        self,
        layer_idx: int,
        current_seq_len: int,
        target_seq_len: Optional[int] = None,
    ) -> EvictionResult:
        """Compute which positions to evict for a given layer.

        Args:
            layer_idx: Layer index.
            current_seq_len: Current number of cached tokens.
            target_seq_len: Target seq len after eviction.
                           Defaults to max_budget - eviction_batch_size.

        Returns:
            EvictionResult with indices to keep and evict.
        """
        cfg = self.config

        if target_seq_len is None:
            target_seq_len = cfg.max_seq_len_budget - cfg.eviction_batch_size

        target_seq_len = max(target_seq_len, cfg.min_seq_len)

        if current_seq_len <= target_seq_len:
            all_indices = torch.arange(current_seq_len)
            return EvictionResult(
                should_evict=False,
                evict_indices=torch.tensor([], dtype=torch.long),
                keep_indices=all_indices,
                num_evicted=0,
                new_seq_len=current_seq_len,
            )

        num_to_evict = current_seq_len - target_seq_len

        if self.scorer is not None:
            evict_indices = self.scorer.get_eviction_candidates(
                layer_idx, current_seq_len, num_to_evict
            )
        else:
            # Fallback: evict oldest non-initial tokens
            start = 4  # protect first few tokens
            end = start + num_to_evict
            end = min(end, current_seq_len)
            evict_indices = torch.arange(start, end)

        # Compute keep indices as complement
        evict_set = set(evict_indices.tolist())
        keep_indices = torch.tensor(
            [i for i in range(current_seq_len) if i not in evict_set],
            dtype=torch.long,
        )

        actual_evicted = len(evict_set)
        self._eviction_count += 1
        self._total_evicted += actual_evicted

        return EvictionResult(
            should_evict=True,
            evict_indices=evict_indices,
            keep_indices=keep_indices,
            num_evicted=actual_evicted,
            new_seq_len=current_seq_len - actual_evicted,
        )

    def apply_eviction(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply eviction by selecting only kept positions.

        Args:
            keys: (batch, heads, seq_len, head_dim)
            values: (batch, heads, seq_len, head_dim)
            keep_indices: (num_keep,) positions to retain

        Returns:
            (new_keys, new_values) with evicted positions removed
        """
        # keys/values shape: (batch, heads, seq_len, head_dim)
        new_keys = keys[:, :, keep_indices, :]
        new_values = values[:, :, keep_indices, :]
        return new_keys, new_values

    @property
    def stats(self) -> dict:
        return {
            "eviction_count": self._eviction_count,
            "total_evicted": self._total_evicted,
        }

    def reset_stats(self):
        self._eviction_count = 0
        self._total_evicted = 0
