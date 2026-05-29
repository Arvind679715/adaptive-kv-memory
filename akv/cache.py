"""Adaptive Hierarchical KV Cache — the core cache manager.

Implements a three-tier memory hierarchy for KV cache:
  - Hot tier:  GPU HBM, full precision (fp16/bf16)
  - Warm tier: GPU HBM, quantized (4-bit or 2-bit)
  - Cold tier: CPU RAM, quantized

Tokens are dynamically promoted/demoted between tiers based on
importance scores from the attention-based scorer.

Drop-in compatible with HuggingFace's DynamicCache API.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from akv.quantizer import KVQuantizer, QuantConfig, QuantizedTensor
from akv.importance import ImportanceScorer, ImportanceConfig
from akv.evictor import AdaptiveEvictor, EvictionConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    hot_budget: int = 1024
    warm_budget: int = 2048
    warm_bits: int = 4
    cold_bits: int = 2
    group_size: int = 128
    enable_cold_tier: bool = True
    eviction_trigger_ratio: float = 0.9
    eviction_batch_size: int = 64
    initial_tokens_protected: int = 4
    recent_tokens_protected: int = 32
    importance_decay: float = 0.3
    n_anchors: int = 16
    scoring_strategy: str = "importance"  # "importance" or "fifo"


class _LayerCache:
    """Per-layer cache holding KV pairs across tiers."""
    __slots__ = ('hot_keys', 'hot_values', 'warm_keys', 'warm_values',
                 'cold_keys', 'cold_values', 'hot_positions', 'warm_positions',
                 'cold_positions', 'seq_len')

    def __init__(self):
        self.hot_keys: Optional[torch.Tensor] = None       # (B, H, S_hot, D)
        self.hot_values: Optional[torch.Tensor] = None
        self.warm_keys: Optional[QuantizedTensor] = None    # quantized
        self.warm_values: Optional[QuantizedTensor] = None
        self.cold_keys: Optional[QuantizedTensor] = None    # quantized, on CPU
        self.cold_values: Optional[QuantizedTensor] = None
        self.hot_positions: Optional[torch.Tensor] = None   # original position indices
        self.warm_positions: Optional[torch.Tensor] = None
        self.cold_positions: Optional[torch.Tensor] = None
        self.seq_len: int = 0

    @property
    def hot_len(self) -> int:
        return self.hot_keys.shape[2] if self.hot_keys is not None else 0

    @property
    def warm_len(self) -> int:
        return len(self.warm_positions) if self.warm_positions is not None else 0

    @property
    def cold_len(self) -> int:
        return len(self.cold_positions) if self.cold_positions is not None else 0


class AdaptiveKVCache:
    """Hierarchical KV cache with adaptive quantization and eviction.

    Implements the HuggingFace DynamicCache interface so it can be
    used as a drop-in replacement. Internally manages three tiers
    of memory with different precision levels.

    Usage:
        cache = AdaptiveKVCache(config)
        # During inference, use as past_key_values:
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._layers: list[_LayerCache] = []

        # Sub-components
        self._quantizer = KVQuantizer(QuantConfig(
            bits=self.config.warm_bits,
            group_size=self.config.group_size,
        ))
        self._cold_quantizer = KVQuantizer(QuantConfig(
            bits=self.config.cold_bits,
            group_size=self.config.group_size,
        ))
        self._scorer = ImportanceScorer(ImportanceConfig(
            decay_factor=self.config.importance_decay,
            initial_tokens_protected=self.config.initial_tokens_protected,
            recent_tokens_protected=self.config.recent_tokens_protected,
        ))
        self._evictor = AdaptiveEvictor(
            EvictionConfig(
                max_seq_len_budget=self.config.hot_budget + self.config.warm_budget,
                eviction_trigger_ratio=self.config.eviction_trigger_ratio,
                eviction_batch_size=self.config.eviction_batch_size,
            ),
            scorer=self._scorer,
        )

        # Stats
        self._update_count = 0
        self._reorganize_count = 0
        self._promotions = 0
        self._demotions = 0

    # ---- DynamicCache-compatible interface ----

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get assembled KV tensors for a layer (hot + dequantized warm)."""
        return self.get_kv(layer_idx)

    def __iter__(self):
        for i in range(len(self._layers)):
            yield self.get_kv(i)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states and return assembled cache for this layer.

        This is the main entry point, called by the model at each layer
        during forward pass. Compatible with HuggingFace's DynamicCache.update().

        Args:
            key_states: (batch, num_heads, new_seq_len, head_dim)
            value_states: same shape
            layer_idx: which layer
            attention_weights: optional attention probs for importance scoring

        Returns:
            (keys, values) — full assembled cache for attention computation
        """
        # Ensure we have enough layer slots
        while len(self._layers) <= layer_idx:
            self._layers.append(_LayerCache())

        layer = self._layers[layer_idx]

        # Update importance scores if attention weights provided
        if attention_weights is not None:
            self._scorer.update(attention_weights, layer_idx)

        # Append to hot tier
        if layer.hot_keys is None:
            layer.hot_keys = key_states
            layer.hot_values = value_states
            new_len = key_states.shape[2]
            layer.hot_positions = torch.arange(new_len, device=key_states.device)
        else:
            layer.hot_keys = torch.cat([layer.hot_keys, key_states], dim=2)
            layer.hot_values = torch.cat([layer.hot_values, value_states], dim=2)
            old_max = layer.hot_positions.max().item() + 1 if layer.hot_positions.numel() > 0 else 0
            new_positions = torch.arange(
                old_max, old_max + key_states.shape[2],
                device=key_states.device,
            )
            layer.hot_positions = torch.cat([layer.hot_positions, new_positions])

        layer.seq_len = layer.hot_len + layer.warm_len + layer.cold_len
        self._update_count += 1

        # Check if we need to reorganize tiers
        if layer.hot_len > self.config.hot_budget:
            self._reorganize_layer(layer_idx)

        # Return assembled cache (hot + dequantized warm for attention)
        return self.get_kv(layer_idx)

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble full KV cache for a layer from all tiers.

        Hot tokens are returned at full precision. Warm tokens are
        dequantized on-the-fly. Cold tokens are excluded from attention
        (they'd need explicit retrieval for long-range access).
        """
        if layer_idx >= len(self._layers):
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)")

        layer = self._layers[layer_idx]
        parts_k, parts_v = [], []

        # Hot tier: full precision
        if layer.hot_keys is not None:
            parts_k.append(layer.hot_keys)
            parts_v.append(layer.hot_values)

        # Warm tier: dequantize on the fly
        if layer.warm_keys is not None:
            warm_k = self._quantizer.dequantize(layer.warm_keys)
            warm_v = self._quantizer.dequantize(layer.warm_values)
            parts_k.append(warm_k)
            parts_v.append(warm_v)

        if not parts_k:
            raise ValueError(f"Layer {layer_idx} has no cached data")

        keys = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        values = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current visible sequence length (hot + warm)."""
        if layer_idx >= len(self._layers):
            return 0
        layer = self._layers[layer_idx]
        return layer.hot_len + layer.warm_len

    def get_max_cache_shape(self) -> Optional[int]:
        if not self._layers:
            return None
        return max(l.seq_len for l in self._layers)

    # ---- Tier management ----

    def _reorganize_layer(self, layer_idx: int):
        """Reorganize a layer's cache into tiers based on importance.

        Called when hot tier exceeds budget. Demotes low-importance
        tokens from hot to warm (quantized), and from warm to cold (CPU).
        """
        layer = self._layers[layer_idx]
        cfg = self.config

        if layer.hot_keys is None:
            return

        hot_len = layer.hot_len

        # Get tier assignments from scorer
        hot_indices, warm_indices, cold_indices = self._scorer.get_tier_assignments(
            layer_idx=layer_idx,
            seq_len=hot_len,
            hot_budget=cfg.hot_budget,
            warm_budget=cfg.warm_budget,
        )

        if warm_indices.numel() == 0 and cold_indices.numel() == 0:
            return  # Nothing to demote

        device = layer.hot_keys.device
        dtype = layer.hot_keys.dtype

        # Extract warm tokens and quantize
        if warm_indices.numel() > 0:
            warm_k = layer.hot_keys[:, :, warm_indices, :]
            warm_v = layer.hot_values[:, :, warm_indices, :]

            # Merge with existing warm tier
            if layer.warm_keys is not None:
                existing_warm_k = self._quantizer.dequantize(layer.warm_keys)
                existing_warm_v = self._quantizer.dequantize(layer.warm_values)
                warm_k = torch.cat([existing_warm_k, warm_k], dim=2)
                warm_v = torch.cat([existing_warm_v, warm_v], dim=2)
                warm_pos = torch.cat([
                    layer.warm_positions,
                    layer.hot_positions[warm_indices],
                ])
            else:
                warm_pos = layer.hot_positions[warm_indices]

            layer.warm_keys = self._quantizer.quantize(warm_k)
            layer.warm_values = self._quantizer.quantize(warm_v)
            layer.warm_positions = warm_pos
            self._demotions += warm_indices.numel()

        # Extract cold tokens and move to CPU
        if cold_indices.numel() > 0 and cfg.enable_cold_tier:
            cold_k = layer.hot_keys[:, :, cold_indices, :].cpu()
            cold_v = layer.hot_values[:, :, cold_indices, :].cpu()

            if layer.cold_keys is not None:
                existing_cold_k = self._cold_quantizer.dequantize(layer.cold_keys)
                existing_cold_v = self._cold_quantizer.dequantize(layer.cold_values)
                cold_k = torch.cat([existing_cold_k, cold_k], dim=2)
                cold_v = torch.cat([existing_cold_v, cold_v], dim=2)
                cold_pos = torch.cat([
                    layer.cold_positions,
                    layer.hot_positions[cold_indices].cpu(),
                ])
            else:
                cold_pos = layer.hot_positions[cold_indices].cpu()

            layer.cold_keys = self._cold_quantizer.quantize(cold_k)
            layer.cold_values = self._cold_quantizer.quantize(cold_v)
            layer.cold_positions = cold_pos
            self._demotions += cold_indices.numel()

        # Keep only hot tokens
        if hot_indices.numel() > 0:
            layer.hot_keys = layer.hot_keys[:, :, hot_indices, :].contiguous()
            layer.hot_values = layer.hot_values[:, :, hot_indices, :].contiguous()
            layer.hot_positions = layer.hot_positions[hot_indices]
        else:
            layer.hot_keys = None
            layer.hot_values = None
            layer.hot_positions = None

        layer.seq_len = layer.hot_len + layer.warm_len + layer.cold_len
        self._reorganize_count += 1

    def promote_tokens(self, layer_idx: int, positions: torch.Tensor):
        """Promote tokens from warm/cold tier back to hot (for retrieval).

        Used when the model needs to attend to a previously-demoted token
        (e.g., during long-range retrieval).
        """
        layer = self._layers[layer_idx]
        if layer.warm_keys is None:
            return

        # Find which warm positions match requested positions
        pos_set = set(positions.tolist())
        warm_pos = layer.warm_positions
        mask = torch.tensor([p.item() in pos_set for p in warm_pos], dtype=torch.bool)

        if not mask.any():
            return

        # Dequantize the matching warm tokens
        warm_k = self._quantizer.dequantize(layer.warm_keys)
        warm_v = self._quantizer.dequantize(layer.warm_values)

        promote_k = warm_k[:, :, mask, :]
        promote_v = warm_v[:, :, mask, :]

        # Add to hot tier
        if layer.hot_keys is not None:
            layer.hot_keys = torch.cat([layer.hot_keys, promote_k.to(layer.hot_keys.device)], dim=2)
            layer.hot_values = torch.cat([layer.hot_values, promote_v.to(layer.hot_values.device)], dim=2)
            layer.hot_positions = torch.cat([layer.hot_positions, warm_pos[mask]])
        else:
            layer.hot_keys = promote_k
            layer.hot_values = promote_v
            layer.hot_positions = warm_pos[mask]

        # Remove from warm tier
        keep_mask = ~mask
        if keep_mask.any():
            remaining_k = warm_k[:, :, keep_mask, :]
            remaining_v = warm_v[:, :, keep_mask, :]
            layer.warm_keys = self._quantizer.quantize(remaining_k)
            layer.warm_values = self._quantizer.quantize(remaining_v)
            layer.warm_positions = warm_pos[keep_mask]
        else:
            layer.warm_keys = None
            layer.warm_values = None
            layer.warm_positions = None

        self._promotions += mask.sum().item()

    # ---- Memory accounting ----

    def memory_usage(self) -> dict:
        """Report memory usage across all tiers in bytes."""
        hot_bytes = 0
        warm_bytes = 0
        cold_bytes = 0

        for layer in self._layers:
            if layer.hot_keys is not None:
                hot_bytes += layer.hot_keys.nbytes + layer.hot_values.nbytes
            if layer.warm_keys is not None:
                warm_bytes += layer.warm_keys.nbytes + layer.warm_values.nbytes
            if layer.cold_keys is not None:
                cold_bytes += layer.cold_keys.nbytes + layer.cold_values.nbytes

        total = hot_bytes + warm_bytes + cold_bytes
        return {
            "hot_mb": hot_bytes / 1e6,
            "warm_mb": warm_bytes / 1e6,
            "cold_mb": cold_bytes / 1e6,
            "total_mb": total / 1e6,
            "num_layers": len(self._layers),
        }

    def tier_summary(self) -> dict:
        """Summary of tokens in each tier across all layers."""
        hot_total = sum(l.hot_len for l in self._layers)
        warm_total = sum(l.warm_len for l in self._layers)
        cold_total = sum(l.cold_len for l in self._layers)
        n = len(self._layers) or 1
        return {
            "hot_tokens_avg": hot_total / n,
            "warm_tokens_avg": warm_total / n,
            "cold_tokens_avg": cold_total / n,
            "reorganizations": self._reorganize_count,
            "promotions": self._promotions,
            "demotions": self._demotions,
        }

    def reset(self):
        """Clear all cached data."""
        self._layers.clear()
        self._scorer.reset()
        self._evictor.reset_stats()
        self._update_count = 0
        self._reorganize_count = 0
        self._promotions = 0
        self._demotions = 0

    @property
    def scorer(self) -> ImportanceScorer:
        return self._scorer

    @property
    def quantizer(self) -> KVQuantizer:
        return self._quantizer
