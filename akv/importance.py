"""Token importance scoring for adaptive KV cache management.

Scores tokens based on accumulated attention patterns to determine which
KV entries are "hot" (frequently attended) vs "cold" (rarely used).
Supports multiple scoring strategies and exponential decay.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ScoringStrategy(str, Enum):
    ATTENTION_ACCUMULATION = "attention_accumulation"
    HEAVY_HITTER = "heavy_hitter"
    RECENCY_WEIGHTED = "recency_weighted"
    HYBRID = "hybrid"
    FIFO = "fifo"  # Evict oldest unprotected tokens first


@dataclass
class ImportanceConfig:
    strategy: ScoringStrategy = ScoringStrategy.HYBRID
    decay_factor: float = 0.95
    recency_weight: float = 0.3
    attention_weight: float = 0.7
    window_size: int = 64
    initial_tokens_protected: int = 4
    recent_tokens_protected: int = 32


class ImportanceScorer:
    """Scores token positions by importance, used to decide eviction/quantization.

    Core insight: tokens that are consistently attended to across many
    decoding steps are "important" and should be kept at high precision
    in fast memory. Tokens that are rarely attended to can be compressed
    or evicted.

    The scorer maintains a running importance estimate per position,
    updated each time new attention weights are observed.
    """

    def __init__(self, config: Optional[ImportanceConfig] = None):
        self.config = config or ImportanceConfig()
        self._scores: dict[int, torch.Tensor] = {}  # layer_idx -> (seq_len,) scores
        self._step_count: int = 0
        self._total_seq_len: int = 0

    def reset(self):
        """Reset all accumulated scores."""
        self._scores.clear()
        self._step_count = 0
        self._total_seq_len = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def update(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int,
    ):
        """Update importance scores with new attention weights.

        Args:
            attention_weights: (batch, num_heads, query_len, kv_len) attention probs.
                              Can be None if attention is not captured.
            layer_idx: Which layer these weights are from.
        """
        if attention_weights is None:
            return

        # Average over batch and heads: (query_len, kv_len)
        avg_attn = attention_weights.float().mean(dim=(0, 1))

        # Sum over query positions to get per-key importance: (kv_len,)
        key_importance = avg_attn.sum(dim=0)

        kv_len = key_importance.shape[0]
        device = key_importance.device

        if layer_idx not in self._scores or self._scores[layer_idx].shape[0] < kv_len:
            # Initialize or expand scores
            old = self._scores.get(layer_idx)
            new_scores = torch.zeros(kv_len, device=device)
            if old is not None:
                new_scores[:old.shape[0]] = old.to(device)
            self._scores[layer_idx] = new_scores

        scores = self._scores[layer_idx]

        # Apply strategy
        cfg = self.config
        if cfg.strategy == ScoringStrategy.ATTENTION_ACCUMULATION:
            scores[:kv_len] = scores[:kv_len] * cfg.decay_factor + key_importance

        elif cfg.strategy == ScoringStrategy.HEAVY_HITTER:
            # Heavy hitter: track max attention received
            scores[:kv_len] = torch.max(scores[:kv_len], key_importance)

        elif cfg.strategy == ScoringStrategy.RECENCY_WEIGHTED:
            # Pure recency: linear decay from end
            recency = torch.linspace(0, 1, kv_len, device=device)
            scores[:kv_len] = recency

        elif cfg.strategy == ScoringStrategy.HYBRID:
            # Combine attention accumulation with recency
            scores[:kv_len] = scores[:kv_len] * cfg.decay_factor + key_importance
            recency = torch.linspace(0, 1, kv_len, device=device)
            scores[:kv_len] = (
                cfg.attention_weight * scores[:kv_len] / scores[:kv_len].max().clamp(min=1e-10)
                + cfg.recency_weight * recency
            )

        elif cfg.strategy == ScoringStrategy.FIFO:
            # FIFO: importance = position index (higher = more recent = more important)
            scores[:kv_len] = torch.arange(kv_len, dtype=torch.float32, device=device)

        self._scores[layer_idx] = scores
        self._total_seq_len = max(self._total_seq_len, kv_len)
        self._step_count += 1

    def get_scores(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get current importance scores for a layer.

        Returns:
            (seq_len,) tensor of importance scores, or None if no scores yet.
        """
        return self._scores.get(layer_idx)

    def get_aggregated_scores(self) -> Optional[torch.Tensor]:
        """Get importance scores averaged across all layers.

        Returns:
            (seq_len,) tensor of mean importance scores across layers.
        """
        if not self._scores:
            return None
        all_scores = []
        max_len = max(s.shape[0] for s in self._scores.values())
        for scores in self._scores.values():
            padded = torch.zeros(max_len, device=scores.device)
            padded[:scores.shape[0]] = scores
            all_scores.append(padded)
        return torch.stack(all_scores).mean(dim=0)

    def get_tier_assignments(
        self,
        layer_idx: int,
        seq_len: int,
        hot_budget: int,
        warm_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign token positions to memory tiers based on importance.

        Args:
            layer_idx: Layer to get assignments for.
            seq_len: Current sequence length.
            hot_budget: Max tokens in hot tier (HBM, full precision).
            warm_budget: Max tokens in warm tier (HBM, quantized).
            Remaining go to cold tier (CPU).

        Returns:
            (hot_indices, warm_indices, cold_indices) — each a 1D tensor
            of position indices.
        """
        scores = self._scores.get(layer_idx)
        cfg = self.config

        if scores is None or seq_len == 0:
            indices = torch.arange(seq_len)
            hot_end = min(hot_budget, seq_len)
            warm_end = min(hot_budget + warm_budget, seq_len)
            return indices[:hot_end], indices[hot_end:warm_end], indices[warm_end:]

        scores = scores[:seq_len].clone()
        device = scores.device

        # Protect initial tokens (system prompt, BOS) — set infinite importance
        n_protect_start = min(cfg.initial_tokens_protected, seq_len)
        scores[:n_protect_start] = float('inf')

        # Protect recent tokens — always hot
        n_protect_end = min(cfg.recent_tokens_protected, seq_len)
        if n_protect_end > 0:
            scores[-n_protect_end:] = float('inf')

        # Sort by importance descending
        sorted_indices = scores.argsort(descending=True)

        # Assign tiers
        total = sorted_indices.shape[0]
        hot_end = min(hot_budget, total)
        warm_end = min(hot_budget + warm_budget, total)

        hot_indices = sorted_indices[:hot_end].sort().values
        warm_indices = sorted_indices[hot_end:warm_end].sort().values
        cold_indices = sorted_indices[warm_end:].sort().values

        return hot_indices, warm_indices, cold_indices

    def get_eviction_candidates(
        self,
        layer_idx: int,
        seq_len: int,
        num_to_evict: int,
    ) -> torch.Tensor:
        """Get indices of least important tokens to evict.

        Args:
            layer_idx: Layer index.
            seq_len: Current sequence length.
            num_to_evict: How many positions to evict.

        Returns:
            (num_to_evict,) tensor of position indices to evict.
        """
        cfg = self.config

        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long)

        # FIFO shortcut: evict oldest unprotected tokens without needing scores
        if cfg.strategy == ScoringStrategy.FIFO:
            n_protect_start = min(cfg.initial_tokens_protected, seq_len)
            n_protect_end = min(cfg.recent_tokens_protected, seq_len)
            # Evictable range: [n_protect_start, seq_len - n_protect_end)
            evictable_start = n_protect_start
            evictable_end = seq_len - n_protect_end
            n_evictable = max(0, evictable_end - evictable_start)
            num_to_evict = min(num_to_evict, n_evictable)
            if num_to_evict <= 0:
                return torch.tensor([], dtype=torch.long)
            # Oldest first
            return torch.arange(evictable_start, evictable_start + num_to_evict, dtype=torch.long)

        scores = self._scores.get(layer_idx)

        if scores is None:
            return torch.tensor([], dtype=torch.long)

        scores = scores[:seq_len].clone()

        # Protect initial and recent tokens
        n_protect_start = min(cfg.initial_tokens_protected, seq_len)
        scores[:n_protect_start] = float('inf')
        n_protect_end = min(cfg.recent_tokens_protected, seq_len)
        if n_protect_end > 0:
            scores[-n_protect_end:] = float('inf')

        # Get least important
        num_to_evict = min(num_to_evict, (scores != float('inf')).sum().item())
        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long)

        _, bottom_indices = scores.topk(num_to_evict, largest=False)
        return bottom_indices.sort().values

    def stats(self) -> dict:
        return {
            "num_layers_tracked": len(self._scores),
            "step_count": self._step_count,
            "total_seq_len": self._total_seq_len,
        }
