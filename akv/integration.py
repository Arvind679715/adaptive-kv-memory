"""HuggingFace Transformers integration — drop-in adaptive KV cache.

Provides AdaptiveCache that implements the transformers Cache interface,
and a monkey-patch function to inject it into any HF model.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from akv.cache import AdaptiveKVCache, CacheConfig
from akv.production_cache import ProductionCache, ProductionCacheConfig

logger = logging.getLogger(__name__)


class HFAdaptiveCache:
    """HuggingFace-compatible wrapper around AdaptiveKVCache.

    Implements the interface expected by transformers models:
    - update(key_states, value_states, layer_idx, cache_kwargs) -> (keys, values)
    - get_seq_length(layer_idx) -> int
    - Iterable over layers
    - Subscriptable by layer index
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self._cache = AdaptiveKVCache(config)
        self.seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV states. Compatible with DynamicCache.update()."""
        cache_kwargs = cache_kwargs or {}
        attention_weights = cache_kwargs.get("attention_weights")

        keys, values = self._cache.update(
            key_states, value_states, layer_idx,
            attention_weights=attention_weights,
        )

        if layer_idx == 0:
            self.seen_tokens += key_states.shape[2]

        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cache.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        return self._cache.get_max_cache_shape()

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, layer_idx: int):
        return self._cache[layer_idx]

    def __iter__(self):
        return iter(self._cache)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search. Applies to hot tier only."""
        for layer in self._cache._layers:
            if layer.hot_keys is not None:
                layer.hot_keys = layer.hot_keys.index_select(0, beam_idx.to(layer.hot_keys.device))
                layer.hot_values = layer.hot_values.index_select(0, beam_idx.to(layer.hot_values.device))

    @property
    def inner_cache(self) -> AdaptiveKVCache:
        return self._cache

    def memory_usage(self) -> dict:
        return self._cache.memory_usage()

    def tier_summary(self) -> dict:
        return self._cache.tier_summary()

    def reset(self):
        self._cache.reset()
        self.seen_tokens = 0


class HFProductionCache:
    """HuggingFace-compatible wrapper around ProductionCache.

    Uses the zero-allocation production cache with TurboQuant warm tier
    and fused attention. This is the recommended cache for serving.

    Usage:
        from akv.integration import HFProductionCache
        from akv.production_cache import ProductionCacheConfig

        config = ProductionCacheConfig(
            num_layers=22, num_heads=4, head_dim=64,
            hot_budget=512, warm_budget=2048, warm_bits=3,
        )
        cache = HFProductionCache(config)
        outputs = model(input_ids=ids, past_key_values=cache, use_cache=True)
    """

    def __init__(self, config: Optional[ProductionCacheConfig] = None):
        self._cache = ProductionCache(config)
        self.seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV states. Compatible with DynamicCache.update()."""
        cache_kwargs = cache_kwargs or {}
        attention_weights = cache_kwargs.get("attention_weights")

        keys, values = self._cache.update(
            key_states, value_states, layer_idx,
            attention_weights=attention_weights,
        )

        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cache.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        cfg = self._cache.config
        return cfg.hot_budget + cfg.warm_budget

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, layer_idx: int):
        return self._cache[layer_idx]

    def __iter__(self):
        for i in range(len(self._cache)):
            yield self._cache[i]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search (not supported in production mode)."""
        raise NotImplementedError("ProductionCache does not support beam search reordering")

    @property
    def inner_cache(self) -> ProductionCache:
        return self._cache

    def memory_usage(self) -> dict:
        return self._cache.memory_usage()

    def tier_summary(self) -> dict:
        usage = self._cache.memory_usage()
        return {
            "hot_tokens": sum(l.hot_len for l in self._cache._layers) // max(len(self._cache._layers), 1),
            "warm_tokens": sum(l.warm_len for l in self._cache._layers) // max(len(self._cache._layers), 1),
            "migrations": usage.get("migrations", 0),
            "total_mb": usage.get("total_mb", 0),
        }

    def fused_attention(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Run fused mixed-precision attention (fast path for decode)."""
        return self._cache.fused_attention(query, layer_idx)

    def reset(self):
        self._cache.reset()
        self.seen_tokens = 0


def patch_model_for_adaptive_cache(model, cache_config: Optional[CacheConfig] = None):
    """Monkey-patch a HuggingFace model to use AdaptiveKVCache.

    This wraps the model's forward method to automatically inject
    an HFAdaptiveCache when past_key_values is not provided.

    Args:
        model: A HuggingFace CausalLM model
        cache_config: Configuration for the adaptive cache

    Returns:
        The patched model (modified in-place)
    """
    original_forward = model.forward

    def patched_forward(*args, **kwargs):
        if kwargs.get("past_key_values") is None and kwargs.get("use_cache", False):
            kwargs["past_key_values"] = HFAdaptiveCache(cache_config)
        return original_forward(*args, **kwargs)

    model.forward = patched_forward
    model._akv_original_forward = original_forward
    model._akv_cache_config = cache_config
    logger.info("Model patched for adaptive KV cache")
    return model


def unpatch_model(model):
    """Remove the adaptive cache patch from a model."""
    if hasattr(model, '_akv_original_forward'):
        model.forward = model._akv_original_forward
        del model._akv_original_forward
        del model._akv_cache_config
        logger.info("Model unpatched")
    return model


def generate_with_adaptive_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    cache_config: Optional[CacheConfig] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    return_stats: bool = False,
    device: Optional[str] = None,
) -> dict:
    """Generate text using adaptive KV cache management.

    Handles the full generation loop with cache management, importance
    scoring, and tier reorganization.

    Args:
        model: HuggingFace CausalLM
        tokenizer: Corresponding tokenizer
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        cache_config: Cache configuration
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        return_stats: If True, include memory/tier stats in output
        device: Device override

    Returns:
        Dict with 'text', 'tokens', 'num_generated', and optionally 'stats'
    """
    import time

    device = device or next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    cache = HFAdaptiveCache(cache_config)
    generated_ids = []

    t_start = time.perf_counter()

    # Prefill
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Use returned cache if model returned a different object
        if past_key_values is not cache:
            cache = past_key_values

        next_token = _sample(logits, temperature, top_p)
        generated_ids.append(next_token.item())

        # Decode loop
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            cache = outputs.past_key_values
            next_token = _sample(logits, temperature, top_p)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    result = {
        "text": text,
        "tokens": generated_ids,
        "num_generated": len(generated_ids),
        "time_ms": elapsed_ms,
        "tokens_per_second": len(generated_ids) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
    }

    if return_stats and isinstance(cache, HFAdaptiveCache):
        result["memory"] = cache.memory_usage()
        result["tiers"] = cache.tier_summary()

    return result


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample a token from logits with temperature and nucleus sampling."""
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
