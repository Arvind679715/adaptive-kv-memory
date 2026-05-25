"""Baseline KV cache methods for fair comparison.

Implements the key competing approaches from recent literature:

1. **FullCache** — standard DynamicCache, no compression (upper bound on quality)
2. **H2O** (Heavy-Hitter Oracle) — Zhang et al. 2023
   Keeps top-k heavy-hitter tokens + recent window, evicts rest.
3. **KIVI** — Liu et al. 2024
   Uniform per-channel 2-bit quantization of full KV cache.
4. **SnapKV** — Li et al. 2024
   Observation-window-based selection: picks tokens that receive
   high attention in a prefix window, then keeps them for generation.
5. **ScissorHands** — Liu et al. 2023
   Persistence-of-importance: tokens important across multiple steps
   are kept; tokens with transient importance are evicted.

All baselines implement a common interface so they can be swapped
into our evaluation framework directly.
"""
from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================
# Common interface
# ============================================================

class BaseKVCache(ABC):
    """Common interface for all KV cache strategies."""

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states. Return full cache for this layer."""
        ...

    @abstractmethod
    def get_seq_length(self, layer_idx: int = 0) -> int:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def memory_bytes(self) -> int:
        """Total memory used by the cache in bytes."""
        ...

    def __len__(self):
        return self.get_seq_length()


# ============================================================
# 1. FullCache — no compression baseline
# ============================================================

class FullCache(BaseKVCache):
    """Standard full-precision KV cache (DynamicCache equivalent).

    Quality upper bound: no compression, no eviction.
    Memory: O(num_layers * seq_len * num_heads * head_dim * 2 bytes)
    """

    def __init__(self):
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}

    def update(self, key_states, value_states, layer_idx, attention_weights=None):
        if layer_idx in self._keys:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states], dim=2)
        else:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states
        return self._keys[layer_idx], self._values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self._keys:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self):
        self._keys.clear()
        self._values.clear()

    def memory_bytes(self):
        total = 0
        for k in self._keys.values():
            total += k.nbytes
        for v in self._values.values():
            total += v.nbytes
        return total


# ============================================================
# 2. H2O — Heavy-Hitter Oracle (Zhang et al., 2023)
# ============================================================

@dataclass
class H2OConfig:
    """Configuration for H2O cache.

    Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
    of Large Language Models" (Zhang et al., NeurIPS 2023)
    """
    budget: int = 1024           # max KV positions to keep
    heavy_hitter_k: int = 512    # number of heavy-hitter slots
    recent_window: int = 512     # size of recent token window
    # budget = heavy_hitter_k + recent_window


class H2OCache(BaseKVCache):
    """Heavy-Hitter Oracle KV cache.

    Strategy: maintain two pools of tokens:
    1. Heavy-hitters: top-k tokens by cumulative attention score
    2. Recent window: last W tokens (sliding window)

    When the cache exceeds budget, evict the least-attended token
    that is NOT in the recent window.

    Strengths: simple, effective, proven on many models.
    Weaknesses: fixed budget split, no compression — evicted tokens are lost.
    """

    def __init__(self, config: Optional[H2OConfig] = None):
        self.config = config or H2OConfig()
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._scores: dict[int, torch.Tensor] = {}  # cumulative attention scores

    def update(self, key_states, value_states, layer_idx, attention_weights=None):
        cfg = self.config
        device = key_states.device

        if layer_idx in self._keys:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states], dim=2)
        else:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states

        seq_len = self._keys[layer_idx].shape[2]

        # Update attention scores
        if attention_weights is not None:
            # attention_weights: (B, H, q_len, kv_len)
            new_importance = attention_weights.float().mean(dim=(0, 1)).sum(dim=0)  # (kv_len,)
            if layer_idx not in self._scores:
                self._scores[layer_idx] = new_importance.to(device)
            else:
                old = self._scores[layer_idx]
                if old.shape[0] < new_importance.shape[0]:
                    expanded = torch.zeros(new_importance.shape[0], device=device)
                    expanded[:old.shape[0]] = old
                    old = expanded
                old[:new_importance.shape[0]] += new_importance.to(device)
                self._scores[layer_idx] = old

        # Evict if over budget
        if seq_len > cfg.budget:
            self._evict(layer_idx)

        return self._keys[layer_idx], self._values[layer_idx]

    def _evict(self, layer_idx: int):
        cfg = self.config
        seq_len = self._keys[layer_idx].shape[2]
        device = self._keys[layer_idx].device

        # Determine which tokens to keep
        if layer_idx in self._scores and self._scores[layer_idx].shape[0] >= seq_len:
            scores = self._scores[layer_idx][:seq_len].clone()
        else:
            scores = torch.zeros(seq_len, device=device)

        # Protect recent window — set to inf
        recent_start = max(0, seq_len - cfg.recent_window)
        scores[recent_start:] = float('inf')

        # Also protect first 4 tokens (BOS, system prompt)
        n_protect = min(4, seq_len)
        scores[:n_protect] = float('inf')

        # Select top-k heavy hitters from non-protected tokens
        k_select = min(cfg.budget, seq_len)
        _, keep_indices = scores.topk(k_select)
        keep_indices = keep_indices.sort().values

        self._keys[layer_idx] = self._keys[layer_idx][:, :, keep_indices, :]
        self._values[layer_idx] = self._values[layer_idx][:, :, keep_indices, :]
        if layer_idx in self._scores:
            self._scores[layer_idx] = self._scores[layer_idx][keep_indices]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self._keys:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self):
        self._keys.clear()
        self._values.clear()
        self._scores.clear()

    def memory_bytes(self):
        total = 0
        for k in self._keys.values():
            total += k.nbytes
        for v in self._values.values():
            total += v.nbytes
        return total


# ============================================================
# 3. KIVI — Uniform KV Quantization (Liu et al., 2024)
# ============================================================

@dataclass
class KIVIConfig:
    """Configuration for KIVI cache.

    Based on: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
    (Liu et al., ICML 2024)
    """
    key_bits: int = 2            # per-channel quantization for keys
    value_bits: int = 2          # per-token quantization for values
    group_size: int = 128        # group size for quantization
    residual_length: int = 128   # keep last N tokens at full precision


class KIVICache(BaseKVCache):
    """KIVI-style uniform quantized KV cache.

    Strategy: quantize ALL cached KV pairs to low-bit (typically 2-bit).
    Keys use per-channel quantization, values use per-token quantization.
    A small residual buffer keeps the most recent tokens at full precision
    (since they haven't been attended to yet and quantization error is highest).

    Strengths: simple, uniform compression, good quality at 2-bit.
    Weaknesses: no importance-awareness — critical and unimportant tokens
    get the same precision. Our approach assigns more bits to important tokens.
    """

    def __init__(self, config: Optional[KIVIConfig] = None):
        self.config = config or KIVIConfig()
        self._quant_keys: dict[int, dict] = {}
        self._quant_values: dict[int, dict] = {}
        self._residual_keys: dict[int, torch.Tensor] = {}
        self._residual_values: dict[int, torch.Tensor] = {}
        self._seq_lens: dict[int, int] = {}

    def update(self, key_states, value_states, layer_idx, attention_weights=None):
        cfg = self.config

        if layer_idx not in self._residual_keys:
            self._residual_keys[layer_idx] = key_states
            self._residual_values[layer_idx] = value_states
            self._seq_lens[layer_idx] = key_states.shape[2]
        else:
            self._residual_keys[layer_idx] = torch.cat(
                [self._residual_keys[layer_idx], key_states], dim=2
            )
            self._residual_values[layer_idx] = torch.cat(
                [self._residual_values[layer_idx], value_states], dim=2
            )
            self._seq_lens[layer_idx] += key_states.shape[2]

        # If residual exceeds limit, quantize overflow
        res_len = self._residual_keys[layer_idx].shape[2]
        if res_len > cfg.residual_length:
            self._quantize_overflow(layer_idx)

        return self._get_full_kv(layer_idx)

    def _quantize_overflow(self, layer_idx: int):
        """Quantize tokens that overflow the residual buffer."""
        cfg = self.config
        res_k = self._residual_keys[layer_idx]
        res_v = self._residual_values[layer_idx]
        res_len = res_k.shape[2]

        # Split: quantize everything except last residual_length tokens
        n_to_quant = res_len - cfg.residual_length
        to_quant_k = res_k[:, :, :n_to_quant, :]
        to_quant_v = res_v[:, :, :n_to_quant, :]

        # Quantize keys (per-channel: quantize along seq_len dimension)
        quant_k = self._quantize_tensor(to_quant_k, cfg.key_bits, cfg.group_size)
        quant_v = self._quantize_tensor(to_quant_v, cfg.value_bits, cfg.group_size)

        # Merge with existing quantized cache
        if layer_idx in self._quant_keys:
            old_k = self._dequantize_tensor(self._quant_keys[layer_idx])
            old_v = self._dequantize_tensor(self._quant_values[layer_idx])
            merged_k = torch.cat([old_k, to_quant_k], dim=2)
            merged_v = torch.cat([old_v, to_quant_v], dim=2)
            quant_k = self._quantize_tensor(merged_k, cfg.key_bits, cfg.group_size)
            quant_v = self._quantize_tensor(merged_v, cfg.value_bits, cfg.group_size)

        self._quant_keys[layer_idx] = quant_k
        self._quant_values[layer_idx] = quant_v

        # Keep only residual
        self._residual_keys[layer_idx] = res_k[:, :, n_to_quant:, :].contiguous()
        self._residual_values[layer_idx] = res_v[:, :, n_to_quant:, :].contiguous()

    def _get_full_kv(self, layer_idx: int):
        """Assemble full KV cache (dequantized + residual)."""
        parts_k, parts_v = [], []

        if layer_idx in self._quant_keys:
            parts_k.append(self._dequantize_tensor(self._quant_keys[layer_idx]))
            parts_v.append(self._dequantize_tensor(self._quant_values[layer_idx]))

        parts_k.append(self._residual_keys[layer_idx])
        parts_v.append(self._residual_values[layer_idx])

        k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
        return k, v

    @staticmethod
    def _quantize_tensor(tensor: torch.Tensor, bits: int, group_size: int) -> dict:
        """Simple per-group asymmetric quantization."""
        shape = tensor.shape
        flat = tensor.float().reshape(-1, tensor.shape[-1])
        rows, cols = flat.shape

        if cols % group_size != 0:
            pad = group_size - cols % group_size
            flat = torch.nn.functional.pad(flat, (0, pad))
            cols = flat.shape[1]

        grouped = flat.reshape(rows, -1, group_size)
        g_min = grouped.amin(dim=-1, keepdim=True)
        g_max = grouped.amax(dim=-1, keepdim=True)
        max_val = (1 << bits) - 1
        scales = (g_max - g_min) / max_val
        scales = scales.clamp(min=1e-10)
        quantized = torch.round((grouped - g_min) / scales).clamp(0, max_val).to(torch.uint8)

        return {
            "data": quantized,
            "scales": scales.squeeze(-1),
            "zeros": g_min.squeeze(-1),
            "shape": shape,
            "bits": bits,
            "group_size": group_size,
        }

    @staticmethod
    def _dequantize_tensor(qdict: dict) -> torch.Tensor:
        """Dequantize back to float."""
        data = qdict["data"].float()
        scales = qdict["scales"].unsqueeze(-1)
        zeros = qdict["zeros"].unsqueeze(-1)
        original_shape = qdict["shape"]
        D = original_shape[-1]

        dequant = data * scales + zeros
        flat = dequant.reshape(dequant.shape[0], -1)[:, :D]
        return flat.reshape(original_shape).to(torch.float16)

    def get_seq_length(self, layer_idx=0):
        return self._seq_lens.get(layer_idx, 0)

    def reset(self):
        self._quant_keys.clear()
        self._quant_values.clear()
        self._residual_keys.clear()
        self._residual_values.clear()
        self._seq_lens.clear()

    def memory_bytes(self):
        total = 0
        for qk in self._quant_keys.values():
            total += qk["data"].nbytes + qk["scales"].nbytes + qk["zeros"].nbytes
        for qv in self._quant_values.values():
            total += qv["data"].nbytes + qv["scales"].nbytes + qv["zeros"].nbytes
        for k in self._residual_keys.values():
            total += k.nbytes
        for v in self._residual_values.values():
            total += v.nbytes
        return total


# ============================================================
# 4. SnapKV — Observation-Window Selection (Li et al., 2024)
# ============================================================

@dataclass
class SnapKVConfig:
    """Configuration for SnapKV cache.

    Based on: "SnapKV: LLM Knows What You are Looking for Before Generation"
    (Li et al., 2024)
    """
    budget: int = 1024           # total KV budget after compression
    observation_window: int = 64 # last N tokens used to determine importance
    kernel_size: int = 5         # pooling kernel for attention pattern smoothing
    initial_tokens_protected: int = 4


class SnapKVCache(BaseKVCache):
    """SnapKV-style KV cache with observation-window selection.

    Strategy: during prefill, use the attention patterns of the last
    `observation_window` tokens to identify which previous tokens are
    important. Keep the top-k important tokens + the observation window.
    During generation, the selected set is fixed.

    Key insight: attention patterns in the observation window strongly
    predict which tokens will be attended to during generation.

    Strengths: one-shot selection is very efficient, no per-step overhead.
    Weaknesses: fixed selection — if importance shifts during generation,
    can't adapt (our approach can). Also no compression — evicted tokens lost.
    """

    def __init__(self, config: Optional[SnapKVConfig] = None):
        self.config = config or SnapKVConfig()
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._compressed: dict[int, bool] = {}

    def update(self, key_states, value_states, layer_idx, attention_weights=None):
        if layer_idx in self._keys:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states], dim=2)
        else:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states
            self._compressed[layer_idx] = False

        seq_len = self._keys[layer_idx].shape[2]

        # Compress at end of prefill (when seq_len exceeds budget and we have attn)
        if (not self._compressed.get(layer_idx, False) and
            seq_len > self.config.budget and
            attention_weights is not None):
            self._compress(layer_idx, attention_weights)

        return self._keys[layer_idx], self._values[layer_idx]

    def _compress(self, layer_idx: int, attention_weights: torch.Tensor):
        """Select important tokens based on observation window attention."""
        cfg = self.config
        k = self._keys[layer_idx]
        v = self._values[layer_idx]
        seq_len = k.shape[2]

        # Use attention from the observation window (last N query positions)
        # attention_weights: (B, H, q_len, kv_len)
        q_len = attention_weights.shape[2]
        obs_start = max(0, q_len - cfg.observation_window)
        obs_attn = attention_weights[:, :, obs_start:, :seq_len]  # (B, H, obs_win, seq_len)

        # Average attention over batch, heads, and observation queries
        importance = obs_attn.float().mean(dim=(0, 1)).sum(dim=0)  # (seq_len,)

        # Smooth with average pooling to capture cluster patterns
        if cfg.kernel_size > 1 and importance.shape[0] > cfg.kernel_size:
            importance_1d = importance.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
            pad = cfg.kernel_size // 2
            importance = torch.nn.functional.avg_pool1d(
                importance_1d, cfg.kernel_size, stride=1, padding=pad,
            ).squeeze()[:seq_len]

        # Protect initial tokens
        n_protect = min(cfg.initial_tokens_protected, seq_len)
        importance[:n_protect] = float('inf')

        # Keep observation window
        obs_window_start = max(0, seq_len - cfg.observation_window)
        importance[obs_window_start:] = float('inf')

        # Select top-k
        n_select = min(cfg.budget, seq_len)
        _, keep_indices = importance.topk(n_select)
        keep_indices = keep_indices.sort().values

        self._keys[layer_idx] = k[:, :, keep_indices, :].contiguous()
        self._values[layer_idx] = v[:, :, keep_indices, :].contiguous()
        self._compressed[layer_idx] = True

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self._keys:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self):
        self._keys.clear()
        self._values.clear()
        self._compressed.clear()

    def memory_bytes(self):
        total = 0
        for k in self._keys.values():
            total += k.nbytes
        for v in self._values.values():
            total += v.nbytes
        return total


# ============================================================
# 5. ScissorHands — Persistence-of-Importance (Liu et al., 2023)
# ============================================================

@dataclass
class ScissorHandsConfig:
    """Configuration for ScissorHands cache.

    Based on: "ScissorHands: Exploiting the Persistence of Importance
    Hypothesis for LLM KV Cache Compression" (Liu et al., NeurIPS 2023)
    """
    budget: int = 1024
    history_window: int = 8      # track importance over last N steps
    persistence_threshold: float = 0.5  # fraction of steps a token must be "important"
    recent_window: int = 64      # always keep recent tokens
    initial_tokens_protected: int = 4


class ScissorHandsCache(BaseKVCache):
    """ScissorHands KV cache with persistence-of-importance eviction.

    Strategy: track whether each token is "important" (top-k by attention)
    across multiple decoding steps. Tokens that are persistently important
    (above threshold fraction of recent steps) are kept. Tokens with
    transient or no importance are evicted.

    Key insight: truly important tokens (punctuation, key entities, etc.)
    are consistently attended to across many steps. Random spikes don't
    indicate lasting importance.

    Strengths: more robust than single-step importance.
    Weaknesses: requires tracking history, adds per-step overhead.
    No compression — evicted tokens are permanently lost.
    """

    def __init__(self, config: Optional[ScissorHandsConfig] = None):
        self.config = config or ScissorHandsConfig()
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._history: dict[int, list[torch.Tensor]] = {}  # per-layer importance history
        self._step: int = 0

    def update(self, key_states, value_states, layer_idx, attention_weights=None):
        cfg = self.config

        if layer_idx in self._keys:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key_states], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value_states], dim=2)
        else:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states
            self._history[layer_idx] = []

        seq_len = self._keys[layer_idx].shape[2]

        # Track importance history
        if attention_weights is not None:
            importance = attention_weights.float().mean(dim=(0, 1)).sum(dim=0)  # (kv_len,)
            # Pad to current seq_len
            if importance.shape[0] < seq_len:
                padded = torch.zeros(seq_len, device=importance.device)
                padded[:importance.shape[0]] = importance
                importance = padded

            # Binarize: mark top-budget tokens as "important this step"
            _, top_indices = importance[:seq_len].topk(min(cfg.budget, seq_len))
            is_important = torch.zeros(seq_len, device=importance.device, dtype=torch.bool)
            is_important[top_indices] = True
            self._history[layer_idx].append(is_important)

            # Keep only recent history
            if len(self._history[layer_idx]) > cfg.history_window:
                self._history[layer_idx] = self._history[layer_idx][-cfg.history_window:]

        # Evict based on persistence
        if seq_len > cfg.budget and len(self._history.get(layer_idx, [])) >= 2:
            self._evict(layer_idx)

        if layer_idx == 0:
            self._step += 1

        return self._keys[layer_idx], self._values[layer_idx]

    def _evict(self, layer_idx: int):
        cfg = self.config
        seq_len = self._keys[layer_idx].shape[2]
        device = self._keys[layer_idx].device
        history = self._history[layer_idx]

        # Compute persistence: fraction of recent steps each token was important
        min_len = min(h.shape[0] for h in history)
        # Truncate all history entries to min_len for stacking
        truncated = [h[:min_len] for h in history]
        stacked = torch.stack(truncated, dim=0).float()  # (steps, min_len)
        persistence = stacked.mean(dim=0)  # (min_len,)

        # Pad to seq_len (new tokens get persistence = 0)
        if persistence.shape[0] < seq_len:
            padded = torch.zeros(seq_len, device=device)
            padded[:persistence.shape[0]] = persistence
            persistence = padded
        else:
            persistence = persistence[:seq_len]

        # Protect recent window and initial tokens
        recent_start = max(0, seq_len - cfg.recent_window)
        persistence[recent_start:] = float('inf')
        n_protect = min(cfg.initial_tokens_protected, seq_len)
        persistence[:n_protect] = float('inf')

        # Keep top-budget by persistence
        n_keep = min(cfg.budget, seq_len)
        _, keep_indices = persistence.topk(n_keep)
        keep_indices = keep_indices.sort().values

        self._keys[layer_idx] = self._keys[layer_idx][:, :, keep_indices, :].contiguous()
        self._values[layer_idx] = self._values[layer_idx][:, :, keep_indices, :].contiguous()

        # Rebuild history for remaining positions
        self._history[layer_idx] = [h[keep_indices[:min_len]] if h.shape[0] >= keep_indices.max() + 1
                                     else h for h in history]

    def get_seq_length(self, layer_idx=0):
        if layer_idx not in self._keys:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self):
        self._keys.clear()
        self._values.clear()
        self._history.clear()
        self._step = 0

    def memory_bytes(self):
        total = 0
        for k in self._keys.values():
            total += k.nbytes
        for v in self._values.values():
            total += v.nbytes
        return total


# ============================================================
# Factory
# ============================================================

def create_baseline(name: str, **kwargs) -> BaseKVCache:
    """Create a baseline KV cache by name.

    Args:
        name: One of 'full', 'h2o', 'kivi', 'snapkv', 'scissorhands'
        **kwargs: Passed to the config constructor

    Returns:
        Configured BaseKVCache instance
    """
    name = name.lower().replace("-", "").replace("_", "")
    if name == "full":
        return FullCache()
    elif name == "h2o":
        return H2OCache(H2OConfig(**kwargs))
    elif name == "kivi":
        return KIVICache(KIVIConfig(**kwargs))
    elif name == "snapkv":
        return SnapKVCache(SnapKVConfig(**kwargs))
    elif name in ("scissorhands", "scissors"):
        return ScissorHandsCache(ScissorHandsConfig(**kwargs))
    else:
        raise ValueError(f"Unknown baseline: {name}. "
                        f"Choose from: full, h2o, kivi, snapkv, scissorhands")
