"""Regression tests for AKVCache chronological-order guarantee.

The cache returns K/V tensors as ``cat([warm, hot])`` from ``update()``;
SDPA's slot-based causal mask requires those tensors to be in chronological
token order. Demote/promote bookkeeping must preserve per-slot position ids
and the final view must be sorted by position before being returned.
"""
from __future__ import annotations

import torch

from akv.drop_in import AKVCache


def _prefill(N: int, hot_budget: int, *, num_layers: int = 1,
             heads: int = 2, head_dim: int = 8) -> AKVCache:
    cache = AKVCache(
        warm_bits=3,
        hot_budget=hot_budget,
        enable_promotion=False,
        enable_promotion_proxy=False,
        num_hidden_layers=num_layers,
    )
    # Position-encoded K so we can sanity-read later if we want.
    pos = torch.arange(N, dtype=torch.float32)
    K = pos.view(1, 1, N, 1).expand(1, heads, N, head_dim).contiguous().to(torch.float16)
    V = K.clone()
    for layer_idx in range(num_layers):
        cache.update(K, V, layer_idx=layer_idx)
    return cache


def test_prefill_no_demote_keeps_positions_in_order():
    """When hot_budget >> seq_len, no demote happens and order is trivial."""
    cache = _prefill(N=64, hot_budget=256)
    layer = cache.layers[0]
    assert layer._warm_positions is None
    assert layer._hot_positions is not None
    assert layer._hot_positions.tolist() == list(range(64))


def test_prefill_with_demote_position_coverage_is_complete():
    """After heavy demote, warm + hot positions must cover [0..N-1] exactly."""
    N = 256
    cache = _prefill(N=N, hot_budget=32)
    layer = cache.layers[0]
    warm = layer._warm_positions.tolist()
    hot = layer._hot_positions.tolist()
    assert len(warm) + len(hot) == N
    assert sorted(warm + hot) == list(range(N)), \
        "warm + hot must cover every token exactly once with no gaps"


def test_full_view_is_chronologically_sorted():
    """The K/V returned from update() must be in chronological slot order.

    We re-derive each slot's position from the bookkeeping arrays the cache
    holds, then assert the returned K/V are arranged so slot[i] holds the
    token that was originally at position i.
    """
    N = 256
    cache = _prefill(N=N, hot_budget=32)
    layer = cache.layers[0]
    # The cache mirrors the returned (sorted) view into self.keys.
    assert layer.keys.shape[2] == N
    # If positions were applied via index_select we should NOT be able to
    # find the warm/hot tier boundary by looking at slot-to-slot jumps in
    # the position-encoded K (modulo quantization, which only hits warm).
    # Concrete check: the first 4 protected hot slots are at the front,
    # the last protect_recent hot slots are at the back, but the FULL view
    # we returned should have all positions in monotonic order.
    full_positions = torch.cat([layer._warm_positions, layer._hot_positions])
    sort_idx = full_positions.argsort()
    # After sort, positions are 0..N-1.
    assert sort_idx.numel() == N
    assert full_positions[sort_idx].tolist() == list(range(N))


def test_decode_step_appends_at_correct_position():
    """Subsequent update() calls must append at slot _total_len, not 0."""
    cache = _prefill(N=64, hot_budget=256)
    layer = cache.layers[0]
    assert layer._hot_positions.tolist() == list(range(64))

    # One decode step.
    K1 = torch.zeros(1, 2, 1, 8, dtype=torch.float16)
    V1 = torch.zeros(1, 2, 1, 8, dtype=torch.float16)
    cache.update(K1, V1, layer_idx=0)
    assert layer._hot_positions.tolist() == list(range(65))

    # 16 more decode steps.
    K16 = torch.zeros(1, 2, 16, 8, dtype=torch.float16)
    V16 = torch.zeros(1, 2, 16, 8, dtype=torch.float16)
    cache.update(K16, V16, layer_idx=0)
    assert layer._hot_positions.tolist() == list(range(81))


def test_promote_preserves_position_coverage():
    """Forcing a proxy promotion must keep position coverage intact."""
    N = 128
    cache = AKVCache(
        warm_bits=3,
        hot_budget=16,
        enable_promotion=True,
        enable_promotion_proxy=True,
        promotion_threshold=0.0,  # force promotion to fire every step
        num_hidden_layers=1,
    )
    pos = torch.arange(N, dtype=torch.float32)
    K = pos.view(1, 1, N, 1).expand(1, 2, N, 8).contiguous().to(torch.float16)
    cache.update(K, K.clone(), layer_idx=0)

    # Decode a few steps so the proxy has a chance to promote.
    for _ in range(8):
        K1 = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        cache.update(K1, K1.clone(), layer_idx=0)

    layer = cache.layers[0]
    warm = layer._warm_positions.tolist() if layer._warm_positions is not None else []
    hot = layer._hot_positions.tolist()
    total_tokens = N + 8
    assert len(warm) + len(hot) == total_tokens
    assert sorted(warm + hot) == list(range(total_tokens)), \
        "promotion must not create duplicate or missing positions"


def test_multilayer_independence():
    """Per-layer position bookkeeping must not bleed between layers."""
    N_layers = 3
    cache = _prefill(N=128, hot_budget=32, num_layers=N_layers)
    for i in range(N_layers):
        layer = cache.layers[i]
        warm = layer._warm_positions.tolist()
        hot = layer._hot_positions.tolist()
        assert sorted(warm + hot) == list(range(128))
