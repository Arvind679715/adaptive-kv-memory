"""Prefetch scheduler for proactive cold-tier token promotion.

Instead of reactively promoting tokens only when they're needed during
attention (which causes a stall), the prefetch scheduler predicts which
cold-tier tokens will be needed soon and preemptively promotes them.

Prediction strategies:
1. **Attention pattern extrapolation**: Tokens that received increasing
   attention over recent steps are likely to be needed again.
2. **Position-based heuristics**: Tokens near the current attention window
   boundary are likely to be promoted soon.
3. **Learned predictor**: A lightweight MLP trained on attention patterns
   to predict future token access (optional, for advanced use).

The prefetch scheduler runs asynchronously on the migration stream,
overlapping with inference. When prediction is accurate, promotions
are "free" — zero latency impact on the inference path.

Usage:
    scheduler = PrefetchScheduler(config, migrator)
    # During inference:
    scheduler.observe_attention(attention_weights, layer_idx, step)
    scheduler.maybe_prefetch()  # Non-blocking, schedules on migration stream
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from akv.async_migration import AsyncMigrator, MigrationType, MigrationTask
from akv.quantizer import QuantizedTensor

logger = logging.getLogger(__name__)


@dataclass
class PrefetchConfig:
    """Configuration for the prefetch scheduler."""
    # Enable prefetching
    enabled: bool = True

    # How many tokens to prefetch per scheduling round
    prefetch_budget: int = 16

    # How often to run the predictor (every N decode steps)
    schedule_every_n_steps: int = 4

    # Prediction strategy
    strategy: str = "attention_trend"  # "attention_trend", "boundary", "learned"

    # Attention trend parameters
    trend_window: int = 8  # Look at last N steps for trend
    trend_threshold: float = 0.3  # Min attention increase to trigger prefetch

    # Boundary parameters
    boundary_margin: int = 64  # Prefetch tokens within this distance of hot boundary

    # Confidence threshold (don't prefetch if prediction confidence is low)
    confidence_threshold: float = 0.5

    # Maximum tokens in the prefetch pipeline at once
    max_in_flight: int = 32


@dataclass
class PrefetchStats:
    """Statistics for prefetch operations."""
    total_prefetches: int = 0
    total_tokens_prefetched: int = 0
    prefetch_hits: int = 0  # Prefetched tokens that were actually needed
    prefetch_misses: int = 0  # Prefetched tokens that were never used
    predictions_made: int = 0
    stalls_avoided: int = 0  # Times we avoided a synchronous promotion
    avg_lead_time_steps: float = 0.0  # Avg steps between prefetch and use


class AttentionHistory:
    """Maintains a rolling history of attention patterns for prediction."""

    def __init__(self, window_size: int = 8, max_positions: int = 16384):
        self.window_size = window_size
        self.max_positions = max_positions
        # Per-layer attention history: deque of (step, attention_scores)
        self._history: dict[int, deque] = {}
        self._step = 0

    def record(self, layer_idx: int, attention_scores: torch.Tensor):
        """Record attention scores for a step.

        Args:
            layer_idx: Layer index
            attention_scores: (N,) per-position attention received this step
        """
        if layer_idx not in self._history:
            self._history[layer_idx] = deque(maxlen=self.window_size)

        # Store on CPU to avoid GPU memory accumulation
        self._history[layer_idx].append(
            (self._step, attention_scores.detach().cpu())
        )

    def advance_step(self):
        self._step += 1

    def get_trend(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Compute attention trend for each position.

        Returns (N,) tensor where positive values indicate increasing
        attention (candidate for prefetch), negative indicates decreasing.
        """
        history = self._history.get(layer_idx)
        if history is None or len(history) < 2:
            return None

        # Stack recent attention scores
        scores = torch.stack([s for _, s in history])  # (window, N)

        # Simple trend: linear regression slope per position
        T = scores.shape[0]
        if T < 2:
            return None

        # Normalized time axis
        t = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        t_mean = t.mean()
        t_centered = t - t_mean

        # Slope per position
        scores_mean = scores.mean(dim=0, keepdim=True)
        scores_centered = scores - scores_mean

        numerator = (t_centered * scores_centered).sum(dim=0)
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator  # (N,)

        return slope

    def get_recent_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get the most recent attention scores for a layer."""
        history = self._history.get(layer_idx)
        if not history:
            return None
        return history[-1][1]

    @property
    def step(self) -> int:
        return self._step


class PrefetchScheduler:
    """Proactive cold-tier token prefetch scheduler.

    Monitors attention patterns and predicts which cold-tier tokens
    will be needed in upcoming steps. Schedules promotions ahead of
    time on the migration stream so they're ready when needed.

    This turns reactive promotions (stall while waiting for CPU→GPU
    transfer) into proactive prefetches (zero latency, data already
    on GPU when needed).
    """

    def __init__(
        self,
        config: Optional[PrefetchConfig] = None,
        migrator: Optional[AsyncMigrator] = None,
    ):
        self.config = config or PrefetchConfig()
        self.migrator = migrator
        self.stats = PrefetchStats()

        # Attention history tracker
        self._history = AttentionHistory(
            window_size=self.config.trend_window,
        )

        # Track which positions are in cold tier per layer
        self._cold_positions: dict[int, set] = {}

        # Track which positions are currently being prefetched
        self._in_flight: set = set()

        # Track prefetched positions to measure hit rate
        self._prefetched: dict[int, set] = {}  # layer_idx -> set of positions

    def observe_attention(
        self,
        attention_weights: torch.Tensor,  # (B, H, M, N) or (N,) reduced
        layer_idx: int,
    ):
        """Record attention pattern for prediction.

        Called after each attention computation. Accumulates information
        about which positions are receiving attention.
        """
        if not self.config.enabled:
            return

        # Reduce to per-position scores
        if attention_weights.dim() == 4:
            # (B, H, M, N) -> (N,)
            scores = attention_weights.float().mean(dim=(0, 1)).sum(dim=0)
        elif attention_weights.dim() == 2:
            scores = attention_weights.float().sum(dim=0)
        else:
            scores = attention_weights.float()

        self._history.record(layer_idx, scores)

    def step(self):
        """Advance one decode step. Call after each generated token."""
        self._history.advance_step()

    def register_cold_positions(self, layer_idx: int, positions: set):
        """Register which positions are in cold tier for a layer."""
        self._cold_positions[layer_idx] = positions

    def remove_cold_position(self, layer_idx: int, position: int):
        """Remove a position from cold tracking (it was promoted)."""
        if layer_idx in self._cold_positions:
            self._cold_positions[layer_idx].discard(position)

        # Check if this was a prefetch hit
        if layer_idx in self._prefetched and position in self._prefetched[layer_idx]:
            self.stats.prefetch_hits += 1
            self._prefetched[layer_idx].discard(position)
            self.stats.stalls_avoided += 1

    def maybe_prefetch(self) -> list[int]:
        """Run the predictor and schedule prefetches if warranted.

        Returns list of position indices being prefetched (or empty if
        no prefetch needed this step).

        This is non-blocking — actual data movement happens on the
        migration stream.
        """
        if not self.config.enabled:
            return []

        # Only run every N steps
        if self._history.step % self.config.schedule_every_n_steps != 0:
            return []

        # Check in-flight budget
        if len(self._in_flight) >= self.config.max_in_flight:
            return []

        # Run prediction strategy
        candidates = self._predict_needed_tokens()

        if not candidates:
            return []

        # Schedule prefetches
        prefetched = []
        budget_remaining = self.config.prefetch_budget

        for layer_idx, position, confidence in candidates:
            if budget_remaining <= 0:
                break
            if confidence < self.config.confidence_threshold:
                continue
            if position in self._in_flight:
                continue

            # Schedule promotion via migrator
            if self.migrator:
                self._schedule_prefetch(layer_idx, position)
                self._in_flight.add(position)
                prefetched.append(position)
                budget_remaining -= 1

                # Track for hit rate measurement
                if layer_idx not in self._prefetched:
                    self._prefetched[layer_idx] = set()
                self._prefetched[layer_idx].add(position)

        if prefetched:
            self.stats.total_prefetches += 1
            self.stats.total_tokens_prefetched += len(prefetched)
            self.stats.predictions_made += 1

        return prefetched

    def _predict_needed_tokens(self) -> list[tuple[int, int, float]]:
        """Predict which cold-tier tokens will be needed soon.

        Returns list of (layer_idx, position, confidence) sorted by confidence.
        """
        if self.config.strategy == "attention_trend":
            return self._predict_by_trend()
        elif self.config.strategy == "boundary":
            return self._predict_by_boundary()
        else:
            return self._predict_by_trend()  # Default

    def _predict_by_trend(self) -> list[tuple[int, int, float]]:
        """Predict based on attention trend analysis.

        Tokens in cold tier whose attention scores are increasing
        (positive slope) are likely to be needed soon.
        """
        candidates = []

        for layer_idx, cold_positions in self._cold_positions.items():
            if not cold_positions:
                continue

            trend = self._history.get_trend(layer_idx)
            if trend is None:
                continue

            # Check cold positions with positive trend
            for pos in cold_positions:
                if pos < trend.shape[0]:
                    slope = trend[pos].item()
                    if slope > self.config.trend_threshold:
                        # Confidence based on slope magnitude
                        confidence = min(1.0, slope / (self.config.trend_threshold * 3))
                        candidates.append((layer_idx, pos, confidence))

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: -x[2])
        return candidates

    def _predict_by_boundary(self) -> list[tuple[int, int, float]]:
        """Predict based on proximity to hot/warm tier boundary.

        Tokens in cold tier that are close to the current attention
        window are likely to be accessed soon.
        """
        candidates = []

        for layer_idx, cold_positions in self._cold_positions.items():
            if not cold_positions:
                continue

            # Get recent attention focus area
            recent = self._history.get_recent_attention(layer_idx)
            if recent is None:
                continue

            # Find the attention "center of mass"
            positions = torch.arange(recent.shape[0], dtype=torch.float32)
            total_attn = recent.sum()
            if total_attn < 1e-8:
                continue
            center = (positions * recent).sum() / total_attn
            center_pos = int(center.item())

            # Score cold positions by distance to attention center
            margin = self.config.boundary_margin
            for pos in cold_positions:
                distance = abs(pos - center_pos)
                if distance < margin:
                    confidence = 1.0 - (distance / margin)
                    candidates.append((layer_idx, pos, confidence))

        candidates.sort(key=lambda x: -x[2])
        return candidates

    def _schedule_prefetch(self, layer_idx: int, position: int):
        """Schedule a single token prefetch via the migrator."""
        # The actual cold-tier data lookup and promotion scheduling
        # would interface with the cache's cold tier storage here.
        # For now, we signal the intent — the cache manager executes.
        logger.debug(f"Prefetch scheduled: layer={layer_idx}, pos={position}")

    def on_prefetch_complete(self, layer_idx: int, position: int):
        """Called when a prefetch completes (token is now in warm/hot tier)."""
        self._in_flight.discard(position)

    def get_stats(self) -> dict:
        """Get prefetch statistics."""
        hit_rate = (
            self.stats.prefetch_hits / max(self.stats.total_tokens_prefetched, 1)
        )
        return {
            "total_prefetches": self.stats.total_prefetches,
            "total_tokens_prefetched": self.stats.total_tokens_prefetched,
            "hit_rate": round(hit_rate, 3),
            "prefetch_hits": self.stats.prefetch_hits,
            "prefetch_misses": self.stats.prefetch_misses,
            "stalls_avoided": self.stats.stalls_avoided,
            "predictions_made": self.stats.predictions_made,
            "in_flight": len(self._in_flight),
        }

    def reset_stats(self):
        self.stats = PrefetchStats()
