"""Bit-exactness regression tests for AKVCache vs DynamicCache.

These tests are the single biggest defence against transformers Cache-API
drift breaking AKV silently. When ``hot_budget`` is large enough that no
demotion happens and promotion is disabled, AKVCache MUST be bit-exact
with stock DynamicCache on greedy generation \u2014 anything else means the
HF integration is broken (wrong mask, wrong KV length, wrong dispatch).

They use a from-scratch tiny Llama so they run fast on CPU and require
no network or HF downloads. If a future transformers bump changes the
CacheLayerMixin contract, these tests will fail loudly here instead of
silently producing 0% on Kaggle.
"""
import pytest
import torch

pytest.importorskip("transformers")

from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from akv.drop_in import AKVCache


@pytest.fixture(scope="module")
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(cfg).eval()
    return cfg, model


def _generate(model, cache, ids, n=8):
    return model.generate(
        ids, max_new_tokens=n, do_sample=False,
        past_key_values=cache, pad_token_id=0,
    )


def test_akv_bit_exact_with_dynamic_when_no_demotion(tiny_llama):
    """AKVCache with hot_budget >> seq_len must match DynamicCache exactly.

    Any divergence indicates a broken HF integration path (wrong
    ``get_mask_sizes``, wrong tensor returned from ``update``, wrong KV
    length seen by the model, etc.).
    """
    cfg, model = tiny_llama
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    with torch.no_grad():
        out_dyn = _generate(model, DynamicCache(), ids)
        akv = AKVCache(
            warm_bits=4, hot_budget=4096,
            num_hidden_layers=cfg.num_hidden_layers,
            enable_promotion=False,
        )
        out_akv = _generate(model, akv, ids)

    assert torch.equal(out_dyn, out_akv), (
        f"AKVCache diverged from DynamicCache despite no demotion / no "
        f"promotion. This indicates a broken HF Cache-API integration.\n"
        f"  DynamicCache: {out_dyn[0].tolist()}\n"
        f"  AKVCache    : {out_akv[0].tolist()}"
    )


def test_akv_generate_runs_under_demotion(tiny_llama):
    """With a tiny hot_budget the cache must still complete generation."""
    cfg, model = tiny_llama
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    with torch.no_grad():
        akv = AKVCache(
            warm_bits=4, hot_budget=4,
            num_hidden_layers=cfg.num_hidden_layers,
            enable_promotion=False,
        )
        out = _generate(model, akv, ids, n=8)

    # 10 prompt tokens + 8 generated = 18
    assert out.shape[1] == 18


def test_akv_layer_implements_cache_layer_mixin_contract():
    """AKVLayer must satisfy the methods transformers >= 4.46 requires."""
    from akv.drop_in import AKVLayer

    required = (
        "update", "get_mask_sizes", "get_seq_length",
        "get_max_cache_shape", "lazy_initialization",
    )
    for name in required:
        assert hasattr(AKVLayer, name), (
            f"AKVLayer missing required CacheLayerMixin method: {name}"
        )

    # Class-level attributes read during mask construction.
    assert AKVLayer.is_sliding is False
    assert AKVLayer.is_compileable is False


def test_akv_cache_get_mask_sizes_modern_signature(tiny_llama):
    """cache.get_mask_sizes(query_length:int, layer_idx:int) must not crash.

    Regression: prior to commit 4691206 the signature was
    ``(cache_position=None, layer_idx=0)`` and crashed with
    ``TypeError: object of type 'int' has no len()`` when called with
    positional ints by transformers 4.46+.
    """
    cfg, model = tiny_llama
    cache = AKVCache(
        warm_bits=4, hot_budget=4096,
        num_hidden_layers=cfg.num_hidden_layers,
        enable_promotion=False,
    )
    # Prime layer 0 so it exists.
    k = torch.randn(1, 2, 5, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 5, 32, dtype=torch.float32)
    cache.update(k, v, layer_idx=0)

    kv_length, kv_offset = cache.get_mask_sizes(3, 0)
    # After update with 5 tokens, a new query of length 3 should see
    # kv_length = 5 + 3 OR 5 (depending on call order in transformers
    # version). Both are valid \u2014 the contract is just "no crash and
    # returns two ints".
    assert isinstance(kv_length, int)
    assert isinstance(kv_offset, int)
    assert kv_offset == 0
    assert kv_length >= 5
