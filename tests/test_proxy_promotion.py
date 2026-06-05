"""Tests for the attention-free promotion proxy and adaptive hot budget.

These cover the FlashAttention compatibility path: when the model's
attention backend never surfaces softmax weights to the cache (the
default in transformers >= 4.45 with FA2/FA3 or SDPA), promotion has
to fire on a key-similarity signal instead.
"""
from __future__ import annotations

import pytest
import torch

from akv.drop_in import AKVCache, AKVLayer


def _decode_step(layer: AKVLayer, n_new: int = 1, seed: int = 0, scale: float = 1.0,
                 H: int = 4, D: int = 8, direction: torch.Tensor | None = None):
    g = torch.Generator().manual_seed(seed)
    if direction is None:
        k = torch.randn(1, H, n_new, D, generator=g) * scale
        v = torch.randn(1, H, n_new, D, generator=g) * scale
    else:
        # Bias new keys toward a chosen direction so they look similar
        # to a specific demoted token.
        noise = torch.randn(1, H, n_new, D, generator=g) * 0.05
        k = direction.expand(1, H, n_new, D).clone() + noise
        v = torch.randn(1, H, n_new, D, generator=g) * scale
    layer.update(k, v, cache_kwargs=None)


def test_proxy_fires_without_attention_weights():
    """Proxy promotion must fire even when cache_kwargs is empty."""
    layer = AKVLayer(
        warm_bits=3,
        hot_budget=8,
        group_size=4,
        enable_promotion=True,
        enable_promotion_proxy=True,
        promotion_threshold=0.0,  # any positive similarity counts
        proxy_decay=0.0,  # fully overwrite each step (sharp signal)
    )

    # Push enough tokens to force demotion: with hot_budget=8 and
    # 16 single-token decode steps, half end up in warm.
    for i in range(16):
        _decode_step(layer, n_new=1, seed=i, H=4, D=8)

    assert layer._warm_len > 0, "warm tier should be populated after demotion"
    assert layer._proxy_score is not None, "proxy must be live without attention_weights"
    assert layer._proxy_score.shape[0] == layer._warm_len, "proxy length must track warm tier"


def test_proxy_disabled_skips_path():
    """With proxy disabled and no attention weights, no promotion should run."""
    layer = AKVLayer(
        warm_bits=3,
        hot_budget=8,
        group_size=4,
        enable_promotion=True,
        enable_promotion_proxy=False,
    )

    for i in range(16):
        _decode_step(layer, n_new=1, seed=i, H=4, D=8)

    # Proxy state must remain None — that's the FA-compatibility regression
    # we're testing against. Behavior matches the pre-proxy implementation.
    assert layer._proxy_score is None


def test_proxy_ranks_similar_warm_slot_highest():
    """The proxy score should rank the most-similar warm token at the top.

    This is the core invariant of attention-free promotion: when a new
    query points along the same direction as a warm-tier token, that
    token's proxy score should beat the random ones.
    """
    layer = AKVLayer(
        warm_bits=4,
        hot_budget=16,
        group_size=4,
        enable_promotion=False,  # we just want to inspect the score, not promote
        enable_promotion_proxy=True,
        proxy_decay=0.0,
    )

    H, D = 4, 16  # use larger D so random cosines are smaller
    g = torch.Generator().manual_seed(7)

    # Fill 4 random tokens (protect-initial window).
    for _ in range(4):
        layer.update(
            torch.randn(1, H, 1, D, generator=g),
            torch.randn(1, H, 1, D, generator=g),
            cache_kwargs=None,
        )

    # Plant a distinctive direction.
    target_dir = torch.randn(1, 1, 1, D, generator=g)
    target_dir = target_dir / target_dir.norm()
    layer.update(
        target_dir.expand(1, H, 1, D).clone(),
        torch.randn(1, H, 1, D, generator=g),
        cache_kwargs=None,
    )

    # Push enough random tokens to force the target into warm.
    for _ in range(40):
        layer.update(
            torch.randn(1, H, 1, D, generator=g),
            torch.randn(1, H, 1, D, generator=g),
            cache_kwargs=None,
        )

    assert layer._warm_len > 0, "warm tier should be populated"

    # Issue a query pointing along the target direction.
    layer.update(
        target_dir.expand(1, H, 1, D).clone() + 0.01 * torch.randn(1, H, 1, D, generator=g),
        torch.randn(1, H, 1, D, generator=g),
        cache_kwargs=None,
    )

    # Inspect: the warm slot most aligned with target_dir should have the
    # highest proxy score.
    assert layer._proxy_score is not None
    wk = layer._warm_keys_fp16.detach().float().mean(dim=(0, 1))  # (warm_len, D)
    wk_norm = wk / (wk.norm(dim=-1, keepdim=True) + 1e-6)
    true_sim_to_target = (wk_norm @ target_dir.view(-1)).abs()

    # The slot with the highest true similarity should also have the
    # highest proxy score (ties OK; we just need monotonic ranking at top).
    top_true_idx = int(true_sim_to_target.argmax())
    top_proxy_idx = int(layer._proxy_score.argmax())
    assert top_proxy_idx == top_true_idx, (
        f"proxy top slot {top_proxy_idx} != true top slot {top_true_idx}; "
        f"true sim={true_sim_to_target[top_true_idx]:.3f}, "
        f"proxy at true={layer._proxy_score[top_true_idx]:.3f}, "
        f"proxy max={layer._proxy_score[top_proxy_idx]:.3f}"
    )


def test_proxy_score_resizes_on_demote():
    """When demote grows the warm tier, _proxy_score must extend in lockstep."""
    layer = AKVLayer(
        warm_bits=3,
        hot_budget=4,
        group_size=4,
        enable_promotion=True,
        enable_promotion_proxy=True,
        promotion_threshold=10.0,  # ridiculously high -> never promote
        proxy_decay=0.5,
    )

    for i in range(8):
        _decode_step(layer, n_new=1, seed=i, H=2, D=8)

    assert layer._warm_len > 0
    assert layer._proxy_score is not None
    assert layer._proxy_score.shape[0] == layer._warm_len


def test_adaptive_hot_budget_scales_with_context():
    """When adaptive_hot_frac > 0, the effective budget should grow."""
    cache_fixed = AKVCache(
        warm_bits=3,
        hot_budget=8,
        group_size=4,
        adaptive_hot_frac=0.0,
    )
    cache_adaptive = AKVCache(
        warm_bits=3,
        hot_budget=8,
        group_size=4,
        adaptive_hot_frac=0.25,
    )

    H, D = 4, 8
    g = torch.Generator().manual_seed(42)
    for step in range(40):
        k = torch.randn(1, H, 1, D, generator=g)
        v = torch.randn(1, H, 1, D, generator=g)
        cache_fixed.update(k.clone(), v.clone(), layer_idx=0)
        cache_adaptive.update(k.clone(), v.clone(), layer_idx=0)

    fixed_layer = cache_fixed.layers[0]
    adaptive_layer = cache_adaptive.layers[0]

    fixed_hot = fixed_layer._hot_keys.shape[2]
    adaptive_hot = adaptive_layer._hot_keys.shape[2]

    assert fixed_hot <= 8 + 1, f"fixed budget violated: {fixed_hot}"
    # With 40 tokens and 0.25 fraction, adaptive budget should be ~10,
    # so hot tier should hold strictly more than the fixed 8-token cap.
    assert adaptive_hot > fixed_hot, (
        f"adaptive budget did not grow: fixed={fixed_hot} adaptive={adaptive_hot}"
    )


def test_proxy_promotion_does_not_break_bit_exact_disabled():
    """With both promotion paths off, AKVCache must behave like before."""
    cache = AKVCache(
        warm_bits=3,
        hot_budget=8,
        group_size=4,
        enable_promotion=False,
        enable_promotion_proxy=False,
    )
    H, D = 4, 8
    g = torch.Generator().manual_seed(7)
    for _ in range(12):
        k = torch.randn(1, H, 1, D, generator=g)
        v = torch.randn(1, H, 1, D, generator=g)
        cache.update(k, v, layer_idx=0)

    layer = cache.layers[0]
    # No proxy state should have been allocated.
    assert layer._proxy_score is None
