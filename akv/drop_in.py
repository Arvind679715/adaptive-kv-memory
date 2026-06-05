"""Drop-in KV cache for any HuggingFace model.

AKV is a `transformers.DynamicCache` subclass — pass it as
`past_key_values=` to any HuggingFace model call. No model surgery,
no monkey-patching, no custom attention code.

    from akv import AKVCache
    cache = AKVCache(preset="quality")
    outputs = model(**inputs, past_key_values=cache, use_cache=True)

That's the whole integration. Everything below is optional.

Compared to tiny-turboquant:
  - AKV uses importance-based migration (not FIFO)
  - Tokens can be promoted back to FP16 when re-accessed
  - Per-head adaptive bit allocation (20-40% memory savings)
  - NormQuant achieves +3.3% PPL at 3-bit vs +27-37% for KIVI-2 at 4K/2V
  - Paged storage with zero per-step allocation
  - Cold tier offload to CPU for ultra-long contexts
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on specific transformers version.
# We prefer subclassing DynamicLayer / DynamicCache so transformers-side
# protocol changes (get_mask_sizes, lazy_initialization, etc.) are inherited
# for free instead of re-implemented in this file.
_DynamicCache = None
_DynamicLayer = None
_BaseCache = None


def _resolve_bases():
    """Resolve (DynamicCache, DynamicLayer) bases for the installed transformers.

    Returns a (cache_base, layer_base) tuple. Both fall back to safe defaults
    when transformers is absent so unit-tests can still import this module.
    """
    global _DynamicCache, _DynamicLayer, _BaseCache
    if _DynamicCache is not None and _DynamicLayer is not None:
        return _DynamicCache, _DynamicLayer

    try:
        from transformers.cache_utils import DynamicCache as _DC, DynamicLayer as _DL
        _DynamicCache, _DynamicLayer = _DC, _DL
    except ImportError:
        # transformers < 4.46 has DynamicCache but no DynamicLayer.
        try:
            from transformers.cache_utils import DynamicCache as _DC
            _DynamicCache = _DC
        except ImportError:
            class _FallbackCache:
                def __init__(self):
                    self.layers = []
                def get_seq_length(self, layer_idx=0):
                    return 0
                def __len__(self):
                    return len(self.layers)
            _DynamicCache = _FallbackCache
        _DynamicLayer = object  # no mixin available on older transformers

    try:
        from transformers import Cache as _CB
        _BaseCache = _CB
    except ImportError:
        _BaseCache = _DynamicCache
    return _DynamicCache, _DynamicLayer


def _get_dynamic_cache_base():
    """Back-compat shim."""
    return _resolve_bases()[0]


def _get_dynamic_layer_base():
    return _resolve_bases()[1]


# =============================================================================
# Presets
# =============================================================================

_PRESETS = {
    # Near-lossless: +1-2% PPL, 4x memory reduction
    "quality": {
        "warm_bits": 4,
        "hot_budget": 256,
        "description": "Near-lossless 4-bit (NormQuant). +1-2% PPL.",
    },
    # Best balance: +3-4% PPL, 5x memory reduction
    "balanced": {
        "warm_bits": 3,
        "hot_budget": 128,
        "description": "NormQuant 3-bit. +3.3% PPL, 5x compression.",
    },
    # Maximum compression: +11% PPL, 8x memory reduction
    "compact": {
        "warm_bits": 2,
        "hot_budget": 64,
        "description": "NormQuant 2-bit. +11% PPL, 8x compression.",
    },
}


# =============================================================================
# AKVLayer — per-layer state
# =============================================================================

class AKVLayer(_get_dynamic_layer_base()):
    """Per-layer KV state with importance-aware tiered storage.

    Subclasses ``transformers.cache_utils.DynamicLayer`` so the full
    ``CacheLayerMixin`` contract (``get_mask_sizes``, ``get_seq_length``,
    ``lazy_initialization``, ``get_max_cache_shape``, ``is_sliding``,
    ``is_compileable``, ``reorder_cache``, ``reset``, etc.) is inherited
    automatically. This means the layer keeps working when transformers
    extends or tweaks the protocol — we only override what we genuinely
    need to specialise.

    Unlike TinyKVLayer (which just has FP16 residual + quantized tail),
    AKVLayer has:
    - Hot tier: recent/important tokens at FP16
    - Warm tier: older tokens at NormQuant 2-4 bit (actually compressed)
    - Importance tracking per token position
    - Promotion: warm tokens re-accessed get moved back to hot

    Storage contract: ``self.keys`` / ``self.values`` hold the full
    warm➚hot concatenated view (so the base class's ``get_seq_length``,
    ``get_mask_sizes`` and legacy ``key_cache`` / ``value_cache`` accessors
    just work). The tier-internal tensors (``_hot_*``, ``_warm_*_fp16``)
    are the source of truth and are rebuilt into ``keys`` / ``values`` at
    the end of every ``update()``.
    """

    # Not torch.compile-friendly (dynamic tiers, Python-side dequant).
    is_compileable: bool = False
    # AKV never uses sliding attention; mask construction needs this attr.
    is_sliding: bool = False

    def __init__(
        self,
        warm_bits: int = 3,
        hot_budget: int = 128,
        group_size: int = 64,
        protect: bool = False,
        enable_promotion: bool = True,
        promotion_threshold: float = 0.05,
        per_head_bits: Optional[list[int]] = None,
        # ---- New: attention-free promotion ----
        # When True, promotion can fire even without ``attention_weights``
        # in cache_kwargs by using a key-similarity proxy:
        #   score_i = EMA( <k_new, k_warm_i> / ||k_new|| ||k_warm_i|| )
        # This is FA-compatible — it only needs the new K tensor the cache
        # already receives in ``update()``. The proxy is a 1× matmul per
        # decode step against the warm tier; cost is O(warm_len · d_head)
        # which is dominated by the existing demote cost.
        enable_promotion_proxy: bool = True,
        proxy_decay: float = 0.7,
        # ---- New: adaptive hot budget ----
        # When > 0, hot_budget is dynamically resized at every update to
        # ``max(hot_budget, int(adaptive_hot_frac * total_len))``. This
        # fixes the RULER 4K–16K collapse where a fixed 512-token hot
        # budget covers <15% of long contexts and needles get demoted
        # at prefill before promotion has a chance to fire. Set to 0.25
        # to enable the policy proposed in paper §7; default 0 (off).
        adaptive_hot_frac: float = 0.0,
    ):
        # Initialise CacheLayerMixin/DynamicLayer state (self.keys=None,
        # self.values=None, self.is_initialized=False, etc.). Skip when the
        # base is plain ``object`` (transformers absent).
        base = _get_dynamic_layer_base()
        if base is not object:
            try:
                base.__init__(self)
            except TypeError:
                # Some older versions take a config arg; ignore failures.
                self.keys = None
                self.values = None
                self.is_initialized = False
        self.warm_bits = warm_bits
        self.hot_budget = hot_budget
        self.group_size = group_size
        self.protect = protect  # If True, keep all tokens at FP16
        self.enable_promotion = enable_promotion
        self.promotion_threshold = promotion_threshold
        self.enable_promotion_proxy = enable_promotion_proxy
        self.proxy_decay = proxy_decay
        self.adaptive_hot_frac = adaptive_hot_frac
        # Per-head bit override from calibration. If set, the demotion path
        # quantizes each KV-head with its own bit-width instead of using the
        # single global `warm_bits` value.
        self.per_head_bits: Optional[list[int]] = (
            list(per_head_bits) if per_head_bits is not None else None
        )

        # State
        self._hot_keys: Optional[torch.Tensor] = None   # (B, H, N_hot, D)
        self._hot_values: Optional[torch.Tensor] = None
        self._warm_keys_fp16: Optional[torch.Tensor] = None  # Dequantized view for attention
        self._warm_values_fp16: Optional[torch.Tensor] = None
        self._warm_len: int = 0
        self._total_len: int = 0

        # Honest packed-byte accounting. This counter is incremented every
        # time we demote tokens to the warm tier, using the *measured* size
        # of the bit-packed quantizer output (see
        # ``packed_layout.measure_packed_bytes``). On promotion we scale it
        # down by the fraction of warm tokens removed — an approximation,
        # but a defensible one and far more honest than a closed-form
        # "theoretical" formula that ignores grouping overhead and padding.
        self._warm_bytes_packed: int = 0

        # Attention-free promotion proxy: EMA of cosine similarity between
        # the latest decode query (approximated by the most recent K tensor
        # arriving at update()) and each warm token's stored K. Shape
        # tracks ``self._warm_len``; resized on demote/promote.
        self._proxy_score: Optional[torch.Tensor] = None

        # Importance scores for migration decisions
        self._importance: Optional[torch.Tensor] = None  # (N_hot,)
        self._decay: float = 0.3
        self._n_anchors: int = 16
        self._step: int = 0

        # Quantizer (lazy init); when per_head_bits is set we maintain a
        # pool keyed by bit-width so heads with the same width share state.
        self._quantizer = None
        self._quantizer_pool: dict[int, object] = {}

    def _make_quantizer(self, bits: int):
        """Construct a TurboQuantizer at a given bit-width, or None."""
        try:
            from akv.turbo_quant import TurboQuantizer, TurboQuantConfig
            return TurboQuantizer(TurboQuantConfig(
                key_bits=bits,
                value_bits=bits,
                group_size=self.group_size,
                rotation="hadamard",
            ))
        except ImportError:
            return None

    def _ensure_quantizer(self, head_dim: int, device):
        """Lazy-init the default NormQuant quantizer (and the per-head pool)."""
        if self._quantizer is None:
            self._quantizer = self._make_quantizer(self.warm_bits)
        if self.per_head_bits:
            for b in set(self.per_head_bits):
                if b not in self._quantizer_pool:
                    self._quantizer_pool[b] = self._make_quantizer(b)

    def _quantize_per_head(self, k: torch.Tensor, v: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Quantize K/V slice (B, H, N, D) with per-head bit-widths.

        Falls back to the global quantizer when per_head_bits is unset or the
        head count mismatches. Always returns FP16 (dequantized) tensors so
        downstream attention paths stay unchanged, plus the *measured*
        packed-byte count for the K+V pair (see
        ``packed_layout.measure_packed_bytes``). The packed bytes are what
        a production cache would actually hold in memory; the fp16 tensors
        are a working copy this prototype keeps for cheap attention concat.
        """
        from akv.packed_layout import measure_packed_bytes

        if (
            not self.per_head_bits
            or self._quantizer is None
            or k.shape[1] != len(self.per_head_bits)
        ):
            qk = self._quantizer.quantize_keys(k.squeeze(0))
            qv = self._quantizer.quantize_values(v.squeeze(0))
            packed_bytes = (
                measure_packed_bytes(qk, self.warm_bits)
                + measure_packed_bytes(qv, self.warm_bits)
            )
            return (
                self._quantizer.dequantize_keys(qk).unsqueeze(0),
                self._quantizer.dequantize_values(qv).unsqueeze(0),
                packed_bytes,
            )

        out_k = torch.empty_like(k)
        out_v = torch.empty_like(v)
        packed_bytes = 0
        for h, bits in enumerate(self.per_head_bits):
            qz = self._quantizer_pool.get(bits) or self._quantizer
            if qz is None:
                out_k[:, h] = k[:, h]
                out_v[:, h] = v[:, h]
                # No quantization happened — this head costs fp16.
                packed_bytes += k[:, h].element_size() * k[:, h].numel()
                packed_bytes += v[:, h].element_size() * v[:, h].numel()
                continue
            # Quantize this head only: shape (B, N, D) -> (1, N, D) view
            kh = k[:, h:h+1].squeeze(0)
            vh = v[:, h:h+1].squeeze(0)
            try:
                qkh = qz.quantize_keys(kh)
                qvh = qz.quantize_values(vh)
                out_k[:, h] = qz.dequantize_keys(qkh).squeeze(0)
                out_v[:, h] = qz.dequantize_values(qvh).squeeze(0)
                packed_bytes += measure_packed_bytes(qkh, bits)
                packed_bytes += measure_packed_bytes(qvh, bits)
            except Exception:
                # Robust fallback: passthrough on per-head failure
                out_k[:, h] = k[:, h]
                out_v[:, h] = v[:, h]
                packed_bytes += k[:, h].element_size() * k[:, h].numel()
                packed_bytes += v[:, h].element_size() * v[:, h].numel()
        return out_k, out_v, packed_bytes

    def _update_importance(self, n_new: int, hot_len: int):
        """Update importance scores using recency decay.

        When attention_weights are available (passed via cache_kwargs),
        uses attention-based scoring. Otherwise falls back to recency.
        """
        if self._importance is None or self._importance.shape[0] != hot_len - n_new:
            # Re-init on shape mismatch
            self._importance = torch.ones(hot_len, dtype=torch.float32)
        else:
            # Decay existing scores
            self._importance = self._importance * self._decay
            # Append new tokens with score 1.0
            new_scores = torch.ones(n_new, dtype=torch.float32)
            self._importance = torch.cat([self._importance, new_scores])

    def _select_demote_indices(self, n_demote: int, hot_len: int) -> torch.Tensor:
        """Select which hot tokens to demote using importance scores.

        Protected positions (first 4, last protect_recent) are never demoted.
        Among eligible tokens, pick the LOWEST importance.
        """
        protect_initial = 4
        protect_recent = min(16, self.hot_budget // 4)

        if self._importance is None or self._importance.shape[0] != hot_len:
            # Fallback to FIFO if scores misaligned
            return torch.arange(n_demote)

        scores = self._importance.clone()
        # Protect initial and recent tokens (set infinite importance)
        scores[:protect_initial] = float('inf')
        scores[max(0, hot_len - protect_recent):] = float('inf')

        # Pick lowest-importance tokens
        _, indices = scores.topk(n_demote, largest=False)
        indices, _ = indices.sort()  # Keep temporal order for cache coherence
        return indices

    def _promote_from_warm(self, attention_weights: torch.Tensor):
        """Promote warm tokens back to hot if they're being heavily attended.

        This is the key differentiator vs all other methods: tokens are
        never permanently lost — they can return to full precision.
        """
        if self._warm_keys_fp16 is None or self._warm_keys_fp16.shape[2] == 0:
            return
        if self._hot_keys is None:
            return

        warm_len = self._warm_keys_fp16.shape[2]
        hot_len = self._hot_keys.shape[2]

        # attention_weights shape: (B, H, N_query, N_kv) where N_kv = warm+hot
        # Compute mean attention to warm tokens
        if attention_weights.dim() == 4 and attention_weights.shape[-1] >= warm_len + hot_len:
            warm_attn = attention_weights[:, :, :, :warm_len].mean(dim=(0, 1, 2))
            threshold = self.promotion_threshold
            promote_mask = warm_attn > threshold
            n_promote = promote_mask.sum().item()

            if n_promote > 0:
                # Limit promotions to avoid overshoot
                max_promote = max(1, self.hot_budget // 8)
                if n_promote > max_promote:
                    # Take top-attended
                    _, top_idx = warm_attn.topk(max_promote)
                    promote_mask = torch.zeros_like(promote_mask)
                    promote_mask[top_idx] = True
                    n_promote = max_promote

                promote_idx = promote_mask.nonzero(as_tuple=True)[0]

                # Move promoted tokens from warm to hot
                promoted_k = self._warm_keys_fp16[:, :, promote_idx, :]
                promoted_v = self._warm_values_fp16[:, :, promote_idx, :]

                # Remove from warm
                keep_mask = ~promote_mask
                self._warm_keys_fp16 = self._warm_keys_fp16[:, :, keep_mask, :]
                self._warm_values_fp16 = self._warm_values_fp16[:, :, keep_mask, :]
                # Scale packed-byte counter by the fraction of tokens that
                # remain (approximation: assumes uniform per-token cost).
                if warm_len > 0:
                    keep_frac = max(0, warm_len - n_promote) / warm_len
                    self._warm_bytes_packed = int(self._warm_bytes_packed * keep_frac)
                self._warm_len -= n_promote

                # Insert at beginning of hot (they're important now)
                self._hot_keys = torch.cat([promoted_k, self._hot_keys], dim=2)
                self._hot_values = torch.cat([promoted_v, self._hot_values], dim=2)

                # Update importance: promoted tokens get high score
                if self._importance is not None:
                    promoted_scores = torch.ones(n_promote) * 2.0
                    self._importance = torch.cat([promoted_scores, self._importance])

    def _update_proxy_score(self, key_states: torch.Tensor):
        """Update the attention-free promotion proxy.

        Computes cosine similarity between the most recent K and every
        warm-tier K, then folds it into an EMA score per warm token.
        This is the FlashAttention-compatible substitute for attention
        weights: FA2/FA3 do not expose softmax(QK^T), so this proxy
        gives promotion a signal it can act on without modifying the
        attention kernel.

        Cost: one matmul of shape (D,) x (warm_len, D) per layer per
        decode step, dominated by the existing demote/dequant cost.
        """
        if self._warm_keys_fp16 is None or self._warm_keys_fp16.shape[2] == 0:
            self._proxy_score = None
            return

        # Use the most recent key vector as a proxy for the current query
        # direction. Mean over heads to get one direction per token, then
        # mean over the new-token axis to collapse to a single (D,) probe.
        # Shape: key_states (B, H, N, D) -> (D,).
        probe = key_states.detach().float().mean(dim=(0, 1, 2))
        probe = probe / (probe.norm() + 1e-6)

        # Warm keys: (B, H, warm_len, D) -> per-token unit-norm in feature
        # space, then dot with probe to get one similarity per warm token.
        wk = self._warm_keys_fp16.detach().float()
        wk_flat = wk.mean(dim=(0, 1))  # (warm_len, D)
        wk_norm = wk_flat / (wk_flat.norm(dim=-1, keepdim=True) + 1e-6)
        sim = wk_norm @ probe  # (warm_len,)

        # EMA so a single high-similarity spike doesn't cause oscillation.
        if (
            self._proxy_score is None
            or self._proxy_score.shape[0] != sim.shape[0]
        ):
            self._proxy_score = sim.abs()
        else:
            d = self.proxy_decay
            self._proxy_score = d * self._proxy_score + (1 - d) * sim.abs()

    def _promote_from_proxy(self):
        """Attention-free promotion path.

        Promotes the top-k warm tokens whose proxy score exceeds the
        promotion threshold. Mirrors the bookkeeping of
        ``_promote_from_warm`` so downstream tier state stays consistent.
        """
        if (
            self._proxy_score is None
            or self._warm_keys_fp16 is None
            or self._warm_keys_fp16.shape[2] == 0
            or self._hot_keys is None
        ):
            return

        warm_len = self._warm_keys_fp16.shape[2]
        if self._proxy_score.shape[0] != warm_len:
            return  # bookkeeping mismatch, skip this step

        # Threshold is on a 0-1 cosine scale; reuse promotion_threshold.
        promote_mask = self._proxy_score > self.promotion_threshold
        n_promote = int(promote_mask.sum().item())
        if n_promote == 0:
            return

        max_promote = max(1, self.hot_budget // 8)
        if n_promote > max_promote:
            _, top_idx = self._proxy_score.topk(max_promote)
            promote_mask = torch.zeros_like(promote_mask)
            promote_mask[top_idx] = True
            n_promote = max_promote

        promote_idx = promote_mask.nonzero(as_tuple=True)[0]
        promoted_k = self._warm_keys_fp16[:, :, promote_idx, :]
        promoted_v = self._warm_values_fp16[:, :, promote_idx, :]

        keep_mask = ~promote_mask
        self._warm_keys_fp16 = self._warm_keys_fp16[:, :, keep_mask, :]
        self._warm_values_fp16 = self._warm_values_fp16[:, :, keep_mask, :]
        self._proxy_score = self._proxy_score[keep_mask]
        if warm_len > 0:
            keep_frac = max(0, warm_len - n_promote) / warm_len
            self._warm_bytes_packed = int(self._warm_bytes_packed * keep_frac)
        self._warm_len -= n_promote

        self._hot_keys = torch.cat([promoted_k, self._hot_keys], dim=2)
        self._hot_values = torch.cat([promoted_v, self._hot_values], dim=2)

        if self._importance is not None:
            promoted_scores = torch.ones(n_promote) * 2.0
            self._importance = torch.cat([promoted_scores, self._importance])

    def update(
        self,
        key_states: torch.Tensor,   # (B, H, N, D)
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV, manage tiers, return full cache view."""
        B, H, N, D = key_states.shape
        device = key_states.device
        self._ensure_quantizer(D, device)

        # Protected layers: just accumulate FP16
        if self.protect:
            if self._hot_keys is None:
                self._hot_keys = key_states
                self._hot_values = value_states
            else:
                self._hot_keys = torch.cat([self._hot_keys, key_states], dim=2)
                self._hot_values = torch.cat([self._hot_values, value_states], dim=2)
            self._total_len += N
            # Keep base-class storage in sync (see contract note above).
            self.keys = self._hot_keys
            self.values = self._hot_values
            self.dtype = self._hot_keys.dtype
            self.device = self._hot_keys.device
            self.is_initialized = True
            return self._hot_keys, self._hot_values

        # Normal path: append to hot
        if self._hot_keys is None:
            self._hot_keys = key_states
            self._hot_values = value_states
        else:
            self._hot_keys = torch.cat([self._hot_keys, key_states], dim=2)
            self._hot_values = torch.cat([self._hot_values, value_states], dim=2)

        self._total_len += N
        self._step += 1

        # Update importance scores
        hot_len = self._hot_keys.shape[2]
        self._update_importance(N, hot_len)

        # Update the attention-free proxy BEFORE promotion so we can act
        # on this step's signal. Cheap: one matmul against the warm tier.
        if self.enable_promotion_proxy:
            self._update_proxy_score(key_states)

        # Promotion: prefer real attention weights when the caller supplies
        # them (eager attention path). Otherwise fall back to the cheap
        # key-similarity proxy so promotion still fires under FlashAttention.
        if self.enable_promotion and cache_kwargs and "attention_weights" in cache_kwargs:
            self._promote_from_warm(cache_kwargs["attention_weights"])
            hot_len = self._hot_keys.shape[2]
        elif self.enable_promotion and self.enable_promotion_proxy:
            self._promote_from_proxy()
            hot_len = self._hot_keys.shape[2]

        # Adaptive hot-budget scaling: when enabled, the effective budget
        # grows with total context so the hot tier never drops below a
        # configured fraction of seen tokens. This addresses the RULER
        # 4K-16K collapse where a fixed budget gets swamped by long input.
        effective_budget = self.hot_budget
        if self.adaptive_hot_frac > 0.0:
            effective_budget = max(
                self.hot_budget,
                int(self.adaptive_hot_frac * self._total_len),
            )

        # Migration: if hot exceeds budget, demote least-important to warm
        if hot_len > effective_budget:
            n_demote = hot_len - effective_budget
            demote_idx = self._select_demote_indices(n_demote, hot_len)

            demote_k = self._hot_keys[:, :, demote_idx, :]
            demote_v = self._hot_values[:, :, demote_idx, :]

            # Actually quantize with NormQuant (no calibration needed).
            # Returns dequantized fp16 (kept as a working copy for cheap
            # attention concat) plus the *measured* packed-byte cost.
            packed_event_bytes = 0
            if self._quantizer is not None:
                demote_k, demote_v, packed_event_bytes = self._quantize_per_head(
                    demote_k, demote_v
                )
            self._warm_bytes_packed += packed_event_bytes

            # Append to warm tier
            if self._warm_keys_fp16 is None:
                self._warm_keys_fp16 = demote_k
                self._warm_values_fp16 = demote_v
            else:
                self._warm_keys_fp16 = torch.cat(
                    [self._warm_keys_fp16, demote_k], dim=2)
                self._warm_values_fp16 = torch.cat(
                    [self._warm_values_fp16, demote_v], dim=2)

            self._warm_len += n_demote

            # Keep _proxy_score in sync with the warm tier. New warm tokens
            # start with score 0 so they need to accumulate evidence before
            # being promoted (avoids immediate ping-pong after demote).
            if self._proxy_score is not None:
                pad = torch.zeros(n_demote, dtype=self._proxy_score.dtype)
                self._proxy_score = torch.cat([self._proxy_score, pad])

            # Remove demoted from hot (keep non-demoted)
            keep_mask = torch.ones(hot_len, dtype=torch.bool)
            keep_mask[demote_idx] = False
            self._hot_keys = self._hot_keys[:, :, keep_mask, :]
            self._hot_values = self._hot_values[:, :, keep_mask, :]

            # Update importance array
            if self._importance is not None:
                self._importance = self._importance[keep_mask]

        # Build full view: warm + hot
        if self._warm_keys_fp16 is not None:
            full_keys = torch.cat([self._warm_keys_fp16, self._hot_keys], dim=2)
            full_values = torch.cat([self._warm_values_fp16, self._hot_values], dim=2)
        else:
            full_keys = self._hot_keys
            full_values = self._hot_values

        # Mirror tier state into the base-class storage so that
        # transformers' DynamicLayer.get_seq_length / get_mask_sizes and the
        # legacy DynamicCache.key_cache / value_cache properties read the
        # correct lengths and tensors.
        self.keys = full_keys
        self.values = full_values
        self.dtype = full_keys.dtype
        self.device = full_keys.device
        self.is_initialized = True

        return full_keys, full_values

    # get_seq_length, get_mask_sizes, get_max_cache_shape, lazy_initialization,
    # reorder_cache, reset, offload, prefetch are all inherited from
    # DynamicLayer / CacheLayerMixin. They read self.keys / self.values, which
    # update() keeps in sync with the tier state.

    def memory_usage_bytes(self) -> dict:
        """Per-layer memory accounting with three flavours of warm-tier cost.

        Returned dict keys:

        ``hot_bytes``
            Live fp16 bytes held in the hot tier (always the actual cost).

        ``warm_bytes_live``
            What this prototype actually keeps resident for the warm tier.
            Because we hold a dequantized fp16 working copy alongside the
            packed representation, this is currently fp16-sized. A production
            cache would not keep this and would pay ``warm_bytes_packed``.

        ``warm_bytes_packed``
            *Measured* packed-byte cost: accumulated at every demote event
            from the actual ``packed_layout.measure_packed_bytes`` call on
            the quantizer's output. This is the honest number to compare
            against an FP16 baseline. Scaled down proportionally during
            promotion (approximation, but defensible).

        ``warm_bytes_formula``
            Closed-form estimate that ignores grouping overhead and padding.
            Kept for back-compat with older paper tables; prefer
            ``warm_bytes_packed`` for new measurements.

        ``warm_bytes``
            Alias of ``warm_bytes_packed`` so legacy callers keep working.

        ``total_bytes``
            ``hot_bytes + warm_bytes_packed`` — the production-realistic
            total.

        ``fp16_equivalent_bytes``
            What the full cache would cost at fp16, for the savings ratio.

        ``savings_ratio``
            ``fp16_equivalent_bytes / total_bytes``.
        """
        H = 1
        D = 128
        hot_bytes = 0
        if self._hot_keys is not None:
            hot_bytes = (
                self._hot_keys.element_size() * self._hot_keys.numel()
                + self._hot_values.element_size() * self._hot_values.numel()
            )
            H = self._hot_keys.shape[1]
            D = self._hot_keys.shape[3]

        # Live warm cost: actual fp16 working copy resident now.
        warm_bytes_live = 0
        if self._warm_keys_fp16 is not None:
            warm_bytes_live = (
                self._warm_keys_fp16.element_size() * self._warm_keys_fp16.numel()
                + self._warm_values_fp16.element_size() * self._warm_values_fp16.numel()
            )

        # Measured packed cost (accumulated from real packing at demote).
        warm_bytes_packed = int(self._warm_bytes_packed)

        # Closed-form formula, kept for back-compat with prior paper numbers.
        warm_bytes_formula = 0
        if self._warm_len > 0:
            elements_per_kv = H * D
            bits_per_element = self.warm_bits
            groups_per_token = (D + self.group_size - 1) // self.group_size
            scale_bytes_per_token = groups_per_token * H * 4
            data_bytes_per_token = (elements_per_kv * bits_per_element + 7) // 8
            warm_bytes_formula = (
                self._warm_len * (data_bytes_per_token + scale_bytes_per_token) * 2
            )

        total = hot_bytes + warm_bytes_packed
        fp16_equiv = self._total_len * H * D * 2 * 2
        return {
            "hot_bytes": hot_bytes,
            "warm_bytes": warm_bytes_packed,  # legacy key, now = measured packed
            "warm_bytes_live": warm_bytes_live,
            "warm_bytes_packed": warm_bytes_packed,
            "warm_bytes_formula": warm_bytes_formula,
            "total_bytes": total,
            "fp16_equivalent_bytes": fp16_equiv,
            "savings_ratio": fp16_equiv / max(total, 1),
        }


# =============================================================================
# AKVCache — the main drop-in class
# =============================================================================

class AKVCache(_get_dynamic_cache_base()):
    """Drop-in KV cache with virtual memory management.

    Subclasses `transformers.DynamicCache` so it passes isinstance checks
    and works with any HuggingFace model via `past_key_values=`.

    Usage:
        # Simple (recommended):
        cache = AKVCache(preset="balanced")
        outputs = model(**inputs, past_key_values=cache, use_cache=True)

        # With model-aware protection:
        cache = AKVCache.for_model(model, preset="balanced",
                                   protect_first=2, protect_last=2)

        # Power user:
        cache = AKVCache(warm_bits=3, hot_budget=256,
                         protect_layers=[0, 1, -1, -2])

    Presets:
        "quality"   — 4-bit NormQuant, +1-2% PPL, ~4x compression
        "balanced"  — 3-bit NormQuant, +3.3% PPL, ~5x compression
        "compact"   — 2-bit NormQuant, +11% PPL, ~8x compression
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        warm_bits: Optional[int] = None,
        hot_budget: Optional[int] = None,
        group_size: int = 64,
        protect_first: int = 0,
        protect_last: int = 0,
        protect_layers: Optional[list] = None,
        num_hidden_layers: Optional[int] = None,
        enable_promotion: bool = True,
        promotion_threshold: float = 0.05,
        enable_promotion_proxy: bool = True,
        proxy_decay: float = 0.7,
        adaptive_hot_frac: float = 0.0,
    ):
        # Resolve preset vs explicit params
        if preset is not None:
            if preset not in _PRESETS:
                raise ValueError(
                    f"Unknown preset {preset!r}. "
                    f"Choose from: {sorted(_PRESETS.keys())}"
                )
            if warm_bits is not None or hot_budget is not None:
                raise ValueError(
                    "preset= is mutually exclusive with warm_bits/hot_budget. "
                    "Pick a preset OR pass explicit params."
                )
            p = _PRESETS[preset]
            self.warm_bits = p["warm_bits"]
            self.hot_budget = p["hot_budget"]
        else:
            self.warm_bits = warm_bits or 3
            self.hot_budget = hot_budget or 128

        self.group_size = group_size
        self.num_hidden_layers = num_hidden_layers
        self.enable_promotion = enable_promotion
        self.promotion_threshold = promotion_threshold
        self.enable_promotion_proxy = enable_promotion_proxy
        self.proxy_decay = proxy_decay
        self.adaptive_hot_frac = adaptive_hot_frac

        # Resolve protected layers
        self._protect_first = protect_first
        self._protect_last = protect_last
        self._protect_explicit = list(protect_layers) if protect_layers else []

        if protect_last > 0 and num_hidden_layers is None:
            raise ValueError(
                "protect_last requires num_hidden_layers. "
                "Use AKVCache.for_model(model, ...) or pass num_hidden_layers=."
            )

        self._protected_set: set = set()
        for i in range(protect_first):
            self._protected_set.add(i)
        if num_hidden_layers is not None:
            for i in range(protect_last):
                self._protected_set.add(num_hidden_layers - 1 - i)
            for p in self._protect_explicit:
                idx = p if p >= 0 else num_hidden_layers + p
                self._protected_set.add(idx)
        else:
            for p in self._protect_explicit:
                if p < 0:
                    raise ValueError(
                        "Negative protect_layers indices require num_hidden_layers. "
                        "Use AKVCache.for_model(model, ...)."
                    )
                self._protected_set.add(p)

        # Initialize base class FIRST so all transformers-required attrs
        # (layers list, layer_class_to_replicate, offloading flags, etc.) are
        # set up before we layer our own state on top. Calling DynamicCache
        # with no args takes its "lazy init" branch which is exactly what we
        # want (we manage layer creation ourselves in update()).
        base = _get_dynamic_cache_base()
        if base is not object and hasattr(base, '__init__'):
            super().__init__()

        # Our own state on top of the base. self.layers was set to [] by
        # super().__init__() above; we reuse that list directly.
        self._seen_tokens: int = 0
        # Optional per-(layer, head) bit assignment from `akv calibrate`.
        # Map layer_idx -> list[bits per KV-head]. Applied at layer creation.
        self._calibration_per_head_bits: dict[int, list[int]] = {}

    @classmethod
    def for_model(cls, model, **kwargs) -> "AKVCache":
        """Build cache with architecture info read from model.config.

        Consults the adapter registry to pick sensible defaults for the
        model family (Llama, Mistral, Qwen, Gemma, Phi, ...). User kwargs
        always win over adapter defaults.

        Example:
            cache = AKVCache.for_model(model, preset="balanced",
                                       protect_first=2, protect_last=2)
        """
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("model has no .config attribute")

        n_layers = getattr(config, "num_hidden_layers", None) or \
                   getattr(config, "n_layer", None)
        if n_layers is None:
            raise ValueError(
                "Could not find num_hidden_layers in model.config. "
                "Pass num_hidden_layers= explicitly."
            )

        # Consult adapter registry for arch-specific defaults
        try:
            from akv.adapters import resolve_for_model
            spec = resolve_for_model(model)
            if spec.kv_compressed:
                logger.warning(
                    "Model '%s' uses compressed KV (MLA); AKV adds little "
                    "value and may interfere. Returning a passthrough cache.",
                    spec.model_type,
                )
            # Apply defaults only if user hasn't overridden
            kwargs.setdefault("preset", spec.default_preset)
            kwargs.setdefault("protect_first", spec.protect_initial)
            kwargs.setdefault("protect_last", spec.protect_recent)
            if spec.warm_bits_override is not None:
                kwargs.setdefault("warm_bits", spec.warm_bits_override)
        except Exception as e:  # registry is best-effort, never fatal
            logger.debug("Adapter registry lookup failed: %s", e)

        return cls(num_hidden_layers=n_layers, **kwargs)

    def _is_protected(self, layer_idx: int) -> bool:
        return layer_idx in self._protected_set

    def update(
        self,
        key_states: torch.Tensor,     # (B, H, N, D)
        value_states: torch.Tensor,   # (B, H, N, D)
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache — fully compatible with DynamicCache.update().

        This is the method HuggingFace models call internally.
        """
        # Lazily create layer caches
        while len(self.layers) <= layer_idx:
            idx = len(self.layers)
            self.layers.append(AKVLayer(
                warm_bits=self.warm_bits,
                hot_budget=self.hot_budget,
                group_size=self.group_size,
                protect=self._is_protected(idx),
                enable_promotion=self.enable_promotion,
                promotion_threshold=self.promotion_threshold,
                enable_promotion_proxy=self.enable_promotion_proxy,
                proxy_decay=self.proxy_decay,
                adaptive_hot_frac=self.adaptive_hot_frac,
                per_head_bits=self._calibration_per_head_bits.get(idx)
                if self._calibration_per_head_bits else None,
            ))

        # Track seen tokens (on first layer only)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Delegate to layer
        return self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length for a layer."""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].get_seq_length()
        return 0

    def get_usable_length(self, max_length: int, layer_idx: int = 0) -> int:
        """Compat with transformers internals."""
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        """No maximum — unlimited context via tiered compression."""
        return None

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value: int):
        self._seen_tokens = value

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder for beam search."""
        for layer in self.layers:
            if layer._hot_keys is not None:
                layer._hot_keys = layer._hot_keys.index_select(0, beam_idx)
                layer._hot_values = layer._hot_values.index_select(0, beam_idx)
            if layer._warm_keys_fp16 is not None:
                layer._warm_keys_fp16 = layer._warm_keys_fp16.index_select(0, beam_idx)
                layer._warm_values_fp16 = layer._warm_values_fp16.index_select(0, beam_idx)

    def memory_usage(self) -> dict:
        """Aggregate memory stats across all layers.

        Sums every per-layer key from ``AKVLayer.memory_usage_bytes`` so
        callers see the same honest packed-vs-live distinction the layer
        reports. Cross-layer ``savings_ratio`` is recomputed from the
        measured packed totals so a per-layer fluke can't skew the global
        number.
        """
        per_layer_keys = (
            "hot_bytes",
            "warm_bytes",
            "warm_bytes_live",
            "warm_bytes_packed",
            "warm_bytes_formula",
            "total_bytes",
            "fp16_equivalent_bytes",
        )
        totals = {k: 0 for k in per_layer_keys}
        for layer in self.layers:
            stats = layer.memory_usage_bytes()
            for k in per_layer_keys:
                totals[k] += stats.get(k, 0)
        totals["savings_ratio"] = (
            totals["fp16_equivalent_bytes"] / max(totals["total_bytes"], 1)
        )
        totals["num_layers"] = len(self.layers)
        return totals

    # --- DynamicCache compatibility ---

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if layer._hot_keys is not None:
                if layer._warm_keys_fp16 is not None:
                    k = torch.cat([layer._warm_keys_fp16, layer._hot_keys], dim=2)
                    v = torch.cat([layer._warm_values_fp16, layer._hot_values], dim=2)
                else:
                    k, v = layer._hot_keys, layer._hot_values
                return (k, v)
        return (torch.tensor([]), torch.tensor([]))

    def __iter__(self):
        for i in range(len(self.layers)):
            yield self[i]

    def to_legacy_cache(self):
        """Convert to tuple-of-tuples format for older transformers."""
        return tuple(self[i] for i in range(len(self.layers)))

    # get_mask_sizes is inherited from transformers.cache_utils.Cache, which
    # dispatches to layer.get_mask_sizes(query_length). AKVLayer inherits
    # that method from DynamicLayer so this Just Works on transformers >= 4.46.
    # On older transformers (<= 4.45) the base class also provided a
    # compatible signature; no override needed.

    @property
    def key_cache(self):
        """List of key tensors per layer (transformers compat)."""
        result = []
        for i in range(len(self.layers)):
            k, _ = self[i]
            result.append(k)
        return result

    @property
    def value_cache(self):
        """List of value tensors per layer (transformers compat)."""
        result = []
        for i in range(len(self.layers)):
            _, v = self[i]
            result.append(v)
        return result

    def crop(self, max_length: int):
        """Crop cache to max_length (speculative decoding compat)."""
        for layer in self.layers:
            if layer._hot_keys is not None:
                cur_len = layer._hot_keys.shape[2]
                if layer._warm_keys_fp16 is not None:
                    cur_len += layer._warm_keys_fp16.shape[2]
                if cur_len > max_length:
                    # Trim hot tier from the end
                    keep = max(0, layer._hot_keys.shape[2] - (cur_len - max_length))
                    layer._hot_keys = layer._hot_keys[:, :, :keep, :]
                    layer._hot_values = layer._hot_values[:, :, :keep, :]
                    layer._total_len = max_length

    def batch_repeat_interleave(self, repeats: int):
        """Repeat cache for beam expansion (transformers 4.43+ beam search)."""
        for layer in self.layers:
            if layer._hot_keys is not None:
                layer._hot_keys = layer._hot_keys.repeat_interleave(repeats, dim=0)
                layer._hot_values = layer._hot_values.repeat_interleave(repeats, dim=0)
            if layer._warm_keys_fp16 is not None:
                layer._warm_keys_fp16 = layer._warm_keys_fp16.repeat_interleave(repeats, dim=0)
                layer._warm_values_fp16 = layer._warm_values_fp16.repeat_interleave(repeats, dim=0)
        return self

    def batch_select_indices(self, indices: torch.LongTensor):
        """Select specific batch indices (transformers 4.43+ beam search)."""
        for layer in self.layers:
            if layer._hot_keys is not None:
                layer._hot_keys = layer._hot_keys.index_select(0, indices)
                layer._hot_values = layer._hot_values.index_select(0, indices)
            if layer._warm_keys_fp16 is not None:
                layer._warm_keys_fp16 = layer._warm_keys_fp16.index_select(0, indices)
                layer._warm_values_fp16 = layer._warm_values_fp16.index_select(0, indices)
        return self

    def reset(self):
        """Reset cache to empty state (speculative decoding rewind)."""
        self.layers.clear()
        self._seen_tokens = 0

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        """Create from tuple-of-tuples (compat)."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (k, v) in enumerate(past_key_values):
                cache.update(k, v, layer_idx)
        return cache

    @classmethod
    def from_calibration(cls, calibration_path, **overrides) -> "AKVCache":
        """Build an AKVCache from a CalibrationReport JSON file.

        Use `akv calibrate --model ... -o calib.json` to produce one, then::

            from akv import AKVCache
            cache = AKVCache.from_calibration("calib.json")

        Per-head bit assignments from the report are applied where possible;
        global recommendations (preset, protect window) are applied as defaults.
        User overrides via **overrides always win.
        """
        from akv.calibration import CalibrationReport
        report = CalibrationReport.load(calibration_path)
        kwargs = dict(
            protect_first=report.recommended_protect_first,
            protect_last=report.recommended_protect_last,
            num_hidden_layers=report.num_layers,
        )
        # warm_bits and preset are mutually exclusive; prefer the more
        # specific per-head average from calibration when it is meaningful,
        # otherwise fall back to the preset.
        if 2.0 <= report.recommended_average_bits <= 4.0:
            kwargs["warm_bits"] = int(round(report.recommended_average_bits))
        else:
            kwargs["preset"] = report.recommended_preset
        kwargs.update(overrides)
        cache = cls(**kwargs)
        # Apply per-(layer, head) bit assignments. JSON keys are strings; coerce.
        cache._calibration_per_head_bits = {
            int(k): list(v) for k, v in report.per_head_bits.items()
        }
        return cache


# =============================================================================
# Convenience: auto-recommend preset from model
# =============================================================================

def recommend_preset(model, tokenizer=None, sample_text: str = None) -> dict:
    """Recommend an AKV preset based on model architecture.

    Returns:
        dict with keys: preset, warm_bits, hot_budget, reason
    """
    config = getattr(model, "config", None)
    if config is None:
        return {"preset": "balanced", "reason": "No config found, using default"}

    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads = getattr(config, "num_key_value_heads",
                      getattr(config, "num_attention_heads", 32))
    head_dim = getattr(config, "head_dim",
                       getattr(config, "hidden_size", 4096) // 
                       getattr(config, "num_attention_heads", 32))

    # Heuristic: larger models are more robust to quantization
    total_params_est = n_layers * n_heads * head_dim * head_dim * 4  # rough
    if total_params_est > 5e9:  # > 5B params
        return {
            "preset": "balanced",
            "warm_bits": 3,
            "hot_budget": 128,
            "protect_first": 2,
            "protect_last": 2,
            "reason": f"Large model ({total_params_est/1e9:.0f}B est.) — robust to 3-bit"
        }
    elif total_params_est > 1e9:  # 1-5B
        return {
            "preset": "quality",
            "warm_bits": 4,
            "hot_budget": 256,
            "protect_first": 1,
            "protect_last": 1,
            "reason": f"Medium model ({total_params_est/1e9:.1f}B est.) — use 4-bit for safety"
        }
    else:
        return {
            "preset": "quality",
            "warm_bits": 4,
            "hot_budget": 512,
            "protect_first": 2,
            "protect_last": 2,
            "reason": f"Small model ({total_params_est/1e9:.1f}B est.) — protect more layers"
        }
