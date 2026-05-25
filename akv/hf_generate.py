"""Enhanced HuggingFace generation integration.

Provides high-level generation utilities that work seamlessly with
HuggingFace's generate() API, text-generation pipelines, and
streaming generation.

Features:
- Drop-in generate() replacement with adaptive cache
- Streaming token generation with cache stats
- Batch generation with per-sequence cache management
- Pipeline integration for HF text-generation-pipeline
- Automatic cache config tuning based on model/hardware

Usage:
    from akv.hf_generate import AdaptiveGenerator, adaptive_pipeline

    # Simple generation
    gen = AdaptiveGenerator(model, tokenizer)
    output = gen.generate("Hello, world!", max_new_tokens=256)

    # Streaming
    for token in gen.stream("Tell me a story", max_new_tokens=512):
        print(token.text, end="", flush=True)

    # Pipeline
    pipe = adaptive_pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
    result = pipe("Once upon a time")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Optional, Union

import torch

from akv.cache import AdaptiveKVCache, CacheConfig
from akv.integration import HFAdaptiveCache, HFProductionCache
from akv.production_cache import ProductionCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from a single generation call."""
    text: str
    tokens: list[int]
    num_generated: int
    time_ms: float
    tokens_per_sec: float
    memory_usage: Optional[dict] = None
    tier_summary: Optional[dict] = None
    prompt_tokens: int = 0


@dataclass
class StreamToken:
    """A single token emitted during streaming generation."""
    token_id: int
    text: str
    cumulative_text: str
    step: int
    time_ms: float  # Time since generation started
    tier_summary: Optional[dict] = None


@dataclass
class GeneratorConfig:
    """Configuration for the adaptive generator."""
    # Cache config
    hot_budget: int = 1024
    warm_budget: int = 2048
    warm_bits: int = 4
    cold_bits: int = 2
    group_size: int = 128
    enable_cold_tier: bool = True

    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = False

    # Performance
    use_cache: bool = True
    report_stats_every: int = 50  # Report tier stats every N tokens in streaming
    use_production_cache: bool = True  # Use ProductionCache with TurboQuant (recommended)

    def to_cache_config(self) -> CacheConfig:
        return CacheConfig(
            hot_budget=self.hot_budget,
            warm_budget=self.warm_budget,
            warm_bits=self.warm_bits,
            cold_bits=self.cold_bits,
            group_size=self.group_size,
            enable_cold_tier=self.enable_cold_tier,
        )

    def to_production_config(self, num_layers: int, num_heads: int, head_dim: int) -> ProductionCacheConfig:
        """Create ProductionCacheConfig from this generator config + model architecture."""
        return ProductionCacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hot_budget=self.hot_budget,
            warm_budget=self.warm_budget,
            warm_bits=self.warm_bits,
            cold_bits=self.cold_bits,
            group_size=self.group_size,
            warm_quantizer="turbo",
        )


class AdaptiveGenerator:
    """High-level text generation with adaptive KV cache.

    Provides a clean API over the adaptive cache for:
    - Single-shot generation
    - Streaming generation
    - Batched generation
    - Automatic hardware-aware configuration
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GeneratorConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GeneratorConfig()
        self.device = device or str(next(model.parameters()).device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._generation_count = 0
        self._total_tokens_generated = 0

        # Detect model architecture for ProductionCache
        self._num_layers = None
        self._num_heads = None
        self._head_dim = None
        self._detect_model_arch()

    def _detect_model_arch(self):
        """Detect model architecture parameters for ProductionCache configuration."""
        try:
            cfg = self.model.config
            self._num_layers = getattr(cfg, "num_hidden_layers", None)
            num_attention_heads = getattr(cfg, "num_attention_heads", None)
            num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
            hidden_size = getattr(cfg, "hidden_size", None)
            if num_attention_heads and hidden_size:
                self._head_dim = hidden_size // num_attention_heads
                self._num_heads = num_kv_heads
        except Exception:
            pass

    def _make_cache(self, cache_config: Optional[CacheConfig] = None):
        """Create the appropriate cache based on config and model architecture."""
        if (
            self.config.use_production_cache
            and self._num_layers is not None
            and self._num_heads is not None
            and self._head_dim is not None
        ):
            prod_config = self.config.to_production_config(
                self._num_layers, self._num_heads, self._head_dim
            )
            prod_config.device = self.device
            logger.info(
                f"Using ProductionCache: {self._num_layers}L, {self._num_heads}H, "
                f"d={self._head_dim}, hot={prod_config.hot_budget}, "
                f"warm={prod_config.warm_budget}@{prod_config.warm_bits}b (TurboQuant)"
            )
            return HFProductionCache(prod_config)
        else:
            cache_cfg = cache_config or self.config.to_cache_config()
            return HFAdaptiveCache(cache_cfg)

    def generate(
        self,
        prompt: Union[str, list[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_stats: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ) -> Union[GenerationOutput, list[GenerationOutput]]:
        """Generate text with adaptive KV cache.

        Args:
            prompt: Input text (or list for batch generation)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Whether to use sampling
            return_stats: Include memory/tier stats
            cache_config: Override cache config

        Returns:
            GenerationOutput or list thereof for batch input
        """
        if isinstance(prompt, list):
            return self._generate_batch(
                prompt, max_new_tokens, temperature, top_p, top_k, do_sample, return_stats, cache_config
            )

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Use ProductionCache (TurboQuant + zero-alloc) when enabled and model arch is known
        cache = self._make_cache(cache_config)

        t_start = time.perf_counter()
        generated_ids = self._generation_loop(
            input_ids, cache, max_new_tokens, temperature, top_p, top_k, do_sample
        )
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        self._generation_count += 1
        self._total_tokens_generated += len(generated_ids)

        output = GenerationOutput(
            text=text,
            tokens=generated_ids,
            num_generated=len(generated_ids),
            time_ms=elapsed_ms,
            tokens_per_sec=len(generated_ids) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            prompt_tokens=input_ids.shape[1],
        )

        if return_stats:
            output.memory_usage = cache.memory_usage()
            output.tier_summary = cache.tier_summary()

        return output

    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> Generator[StreamToken, None, None]:
        """Stream tokens one-by-one with adaptive KV cache.

        Yields StreamToken objects as they're generated. Useful for
        interactive applications and real-time display.

        Usage:
            for token in gen.stream("Tell me a story"):
                print(token.text, end="", flush=True)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        cache = self._make_cache(cache_config)

        t_start = time.perf_counter()
        cumulative_ids = []

        with torch.inference_mode():
            # Prefill
            outputs = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            past_kv = outputs.past_key_values
            next_token_id = self._sample_token(
                outputs.logits[:, -1, :], temperature, top_p, top_k, do_sample
            )
            cumulative_ids.append(next_token_id)

            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            cumulative_text = token_text

            yield StreamToken(
                token_id=next_token_id,
                text=token_text,
                cumulative_text=cumulative_text,
                step=0,
                time_ms=(time.perf_counter() - t_start) * 1000,
                tier_summary=cache.tier_summary() if isinstance(cache, HFAdaptiveCache) else None,
            )

            # Decode loop
            for step in range(1, max_new_tokens):
                next_input = torch.tensor([[next_token_id]], device=self.device)
                outputs = self.model(input_ids=next_input, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                next_token_id = self._sample_token(
                    outputs.logits[:, -1, :], temperature, top_p, top_k, do_sample
                )

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                cumulative_ids.append(next_token_id)
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                cumulative_text = self.tokenizer.decode(cumulative_ids, skip_special_tokens=True)

                tier_info = None
                if step % self.config.report_stats_every == 0:
                    if isinstance(past_kv, HFAdaptiveCache):
                        tier_info = past_kv.tier_summary()

                yield StreamToken(
                    token_id=next_token_id,
                    text=token_text,
                    cumulative_text=cumulative_text,
                    step=step,
                    time_ms=(time.perf_counter() - t_start) * 1000,
                    tier_summary=tier_info,
                )

        self._generation_count += 1
        self._total_tokens_generated += len(cumulative_ids)

    def _generate_batch(
        self, prompts, max_new_tokens, temperature, top_p, top_k, do_sample, return_stats, cache_config
    ) -> list[GenerationOutput]:
        """Generate for a batch of prompts (sequential, each with own cache)."""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt, max_new_tokens, temperature, top_p, top_k, do_sample, return_stats, cache_config
            )
            results.append(result)
        return results

    def _generation_loop(
        self,
        input_ids: torch.Tensor,
        cache: HFAdaptiveCache,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> list[int]:
        """Core generation loop."""
        generated_ids = []

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            past_kv = outputs.past_key_values
            next_token_id = self._sample_token(
                outputs.logits[:, -1, :], temperature, top_p, top_k, do_sample
            )
            generated_ids.append(next_token_id)

            for _ in range(max_new_tokens - 1):
                next_input = torch.tensor([[next_token_id]], device=self.device)
                outputs = self.model(input_ids=next_input, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                next_token_id = self._sample_token(
                    outputs.logits[:, -1, :], temperature, top_p, top_k, do_sample
                )
                generated_ids.append(next_token_id)
                if next_token_id == self.tokenizer.eos_token_id:
                    break

        return generated_ids

    def _sample_token(
        self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, do_sample: bool
    ) -> int:
        """Sample or greedily select next token."""
        if not do_sample or temperature <= 0:
            return logits.argmax(dim=-1).item()

        logits = logits.float() / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, top_k_indices, top_k_logits)

        # Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @property
    def stats(self) -> dict:
        return {
            "total_generations": self._generation_count,
            "total_tokens_generated": self._total_tokens_generated,
        }


# ============================================================
# Pipeline Integration
# ============================================================

def adaptive_pipeline(
    task: str = "text-generation",
    model: Optional[str] = None,
    cache_config: Optional[CacheConfig] = None,
    device: Optional[str] = None,
    **kwargs,
):
    """Create a HuggingFace pipeline with adaptive KV cache.

    Drop-in replacement for `transformers.pipeline()` that uses
    our adaptive cache for memory-efficient long-context generation.

    Usage:
        pipe = adaptive_pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
        result = pipe("Once upon a time", max_new_tokens=512)
    """
    from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer
    from akv.integration import patch_model_for_adaptive_cache

    # Load model with patched cache
    if model:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model, device_map=device or "auto", **kwargs
        )
        patch_model_for_adaptive_cache(model_obj, cache_config)

        return hf_pipeline(
            task,
            model=model_obj,
            tokenizer=tokenizer,
            device_map=device or "auto",
        )
    else:
        raise ValueError("model must be specified")


# ============================================================
# Auto-Configuration
# ============================================================

def auto_configure(
    model_name: str,
    available_vram_gb: Optional[float] = None,
    target_context_length: int = 8192,
    quality_priority: float = 0.7,  # 0.0 = max compression, 1.0 = max quality
) -> GeneratorConfig:
    """Automatically configure the adaptive cache based on model and hardware.

    Analyzes:
    - Model size (layers, heads, head_dim)
    - Available VRAM
    - Target context length
    - User's quality/memory trade-off preference

    Returns an optimized GeneratorConfig.
    """
    # Estimate model params from name
    model_params = _estimate_model_params(model_name)

    if available_vram_gb is None:
        if torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            available_vram_gb = 16.0  # conservative default

    # Calculate KV cache memory requirements
    num_layers = model_params["num_layers"]
    num_heads = model_params["num_heads"]
    head_dim = model_params["head_dim"]

    # Full cache at target context: layers * 2 * seq_len * heads * head_dim * 2 bytes
    full_cache_gb = (num_layers * 2 * target_context_length * num_heads * head_dim * 2) / 1e9

    # How much VRAM is available for cache (model takes ~50-70%)
    cache_budget_gb = available_vram_gb * 0.3  # 30% for KV cache

    # Configure tiers based on budget
    if cache_budget_gb >= full_cache_gb:
        # Enough VRAM for full cache — use large hot budget
        hot_budget = min(target_context_length, 4096)
        warm_budget = target_context_length - hot_budget
        warm_bits = 4
    elif cache_budget_gb >= full_cache_gb * 0.5:
        # Moderate budget — balanced tiers
        hot_budget = min(1024, target_context_length // 4)
        warm_budget = min(2048, target_context_length // 2)
        warm_bits = 4
    else:
        # Tight budget — aggressive quantization
        hot_budget = min(512, target_context_length // 8)
        warm_budget = min(4096, target_context_length // 2)
        warm_bits = 2

    # Adjust by quality priority
    if quality_priority > 0.8:
        hot_budget = int(hot_budget * 1.5)
        warm_bits = max(warm_bits, 4)
    elif quality_priority < 0.3:
        hot_budget = int(hot_budget * 0.5)
        warm_bits = 2

    config = GeneratorConfig(
        hot_budget=hot_budget,
        warm_budget=warm_budget,
        warm_bits=warm_bits,
        cold_bits=2,
        group_size=128,
        enable_cold_tier=True,
    )

    logger.info(f"Auto-configured: hot={hot_budget}, warm={warm_budget} ({warm_bits}bit), "
                f"cache_budget={cache_budget_gb:.1f}GB, full_cache={full_cache_gb:.1f}GB")

    return config


def _estimate_model_params(model_name: str) -> dict:
    """Estimate model architecture params from name."""
    name_lower = model_name.lower()

    # Common model configurations
    if "70b" in name_lower:
        return {"num_layers": 80, "num_heads": 64, "head_dim": 128}
    elif "34b" in name_lower or "33b" in name_lower:
        return {"num_layers": 60, "num_heads": 56, "head_dim": 128}
    elif "13b" in name_lower:
        return {"num_layers": 40, "num_heads": 40, "head_dim": 128}
    elif "7b" in name_lower or "8b" in name_lower:
        return {"num_layers": 32, "num_heads": 32, "head_dim": 128}
    elif "3b" in name_lower:
        return {"num_layers": 26, "num_heads": 32, "head_dim": 100}
    elif "1b" in name_lower or "1.5b" in name_lower:
        return {"num_layers": 24, "num_heads": 16, "head_dim": 128}
    else:
        # Default to 7B-class
        return {"num_layers": 32, "num_heads": 32, "head_dim": 128}
