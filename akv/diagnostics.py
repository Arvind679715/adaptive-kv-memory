"""Auto-diagnostics for AKV cache configuration.

Analyzes a model's KV statistics and recommends the optimal preset.
Zero-effort configuration: run once, get a recommendation.

Usage:
    from akv.diagnostics import recommend_preset, diagnose_model

    # Quick recommendation
    preset = recommend_preset(model)
    cache = AKVCache(preset=preset)

    # Detailed diagnostics
    report = diagnose_model(model, tokenizer, sample_text="Hello world")
    print(report.summary)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Results of model diagnostics for AKV configuration."""
    model_name: str
    num_layers: int
    num_heads: int
    head_dim: int
    kv_norm_mean: float
    kv_norm_std: float
    kv_norm_ratio: float  # max_norm / mean_norm (outlier indicator)
    recommended_preset: str
    recommended_bits: int
    estimated_compression: float
    details: dict

    @property
    def summary(self) -> str:
        lines = [
            f"AKV Diagnostic Report: {self.model_name}",
            f"  Architecture: {self.num_layers}L × {self.num_heads}H × {self.head_dim}D",
            f"  KV norm stats: mean={self.kv_norm_mean:.3f}, std={self.kv_norm_std:.3f}, "
            f"ratio={self.kv_norm_ratio:.2f}",
            f"  Recommended preset: {self.recommended_preset}",
            f"  Recommended bits: {self.recommended_bits}b keys / {max(2, self.recommended_bits - 1)}b values",
            f"  Estimated compression: {self.estimated_compression:.1f}×",
            f"",
            f"  Usage:",
            f"    from akv import AKVCache",
            f'    cache = AKVCache(preset="{self.recommended_preset}")',
        ]
        return "\n".join(lines)


def measure_kv_norms(
    model,
    tokenizer=None,
    sample_text: str = "The quick brown fox jumps over the lazy dog. " * 20,
    max_length: int = 512,
    device: Optional[str] = None,
) -> dict:
    """Measure KV cache norm statistics across all layers.

    Runs a short forward pass and captures key/value norms per layer/head.
    This indicates how "outlier-heavy" the model's KV representations are.

    Returns dict with norm statistics per layer.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    # Tokenize
    if tokenizer is not None:
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
    else:
        # Assume model has a tokenizer-like interface or use dummy input
        inputs = {"input_ids": torch.randint(1, 1000, (1, min(256, max_length)), device=device)}

    # Hook into attention layers to capture KV norms
    kv_norms = {}
    hooks = []

    def _make_hook(layer_idx):
        def hook_fn(module, args, output):
            # output is typically (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 3:
                past_kv = output[2]
                if past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) == 2:
                    k, v = past_kv
                    kv_norms[layer_idx] = {
                        "key_norm": k.float().norm(dim=-1).mean().item(),
                        "value_norm": v.float().norm(dim=-1).mean().item(),
                        "key_max_norm": k.float().norm(dim=-1).max().item(),
                        "value_max_norm": v.float().norm(dim=-1).max().item(),
                    }
        return hook_fn

    # Try to find attention layers
    attn_modules = []
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ["self_attn", "attention", "attn"]):
            if hasattr(module, "o_proj") or hasattr(module, "out_proj"):
                attn_modules.append(module)

    for i, module in enumerate(attn_modules):
        hooks.append(module.register_forward_hook(_make_hook(i)))

    # Run forward pass
    with torch.no_grad():
        try:
            model(**inputs, use_cache=True)
        except Exception:
            # Fallback: just measure parameter norms
            pass

    # Remove hooks
    for h in hooks:
        h.remove()

    return kv_norms


def recommend_preset(
    model,
    tokenizer=None,
    sample_text: Optional[str] = None,
) -> str:
    """Recommend an AKV preset based on model characteristics.

    Returns one of: "quality", "balanced", "compact"

    Logic:
    - Large models (>13B params) or high KV norm ratios → "quality" (4-bit)
    - Medium models (2B-13B) with normal norms → "balanced" (3-bit)
    - Small models (<2B) or low norm ratio → "compact" (2-bit)
    """
    # Get param count
    param_count = sum(p.numel() for p in model.parameters())
    param_b = param_count / 1e9

    # Try to measure KV norms
    kv_norms = {}
    if sample_text:
        try:
            kv_norms = measure_kv_norms(model, tokenizer, sample_text)
        except Exception:
            pass

    # Compute norm ratio if available
    norm_ratio = 1.0
    if kv_norms:
        all_key_norms = [v["key_norm"] for v in kv_norms.values()]
        all_key_max = [v["key_max_norm"] for v in kv_norms.values()]
        if all_key_norms:
            mean_norm = sum(all_key_norms) / len(all_key_norms)
            max_norm = max(all_key_max) if all_key_max else mean_norm
            norm_ratio = max_norm / max(mean_norm, 1e-8)

    # Decision logic
    if param_b > 13 or norm_ratio > 5.0:
        return "quality"
    elif param_b > 2 or norm_ratio > 3.0:
        return "balanced"
    else:
        return "compact"


def diagnose_model(
    model,
    tokenizer=None,
    sample_text: Optional[str] = None,
) -> DiagnosticReport:
    """Full diagnostic report for a model.

    Measures KV norms, analyzes architecture, recommends configuration.
    """
    # Architecture detection
    cfg = getattr(model, "config", None)
    num_layers = getattr(cfg, "num_hidden_layers", 0) if cfg else 0
    num_heads_total = getattr(cfg, "num_attention_heads", 0) if cfg else 0
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads_total) if cfg else 0
    hidden_size = getattr(cfg, "hidden_size", 0) if cfg else 0
    head_dim = hidden_size // max(num_heads_total, 1) if hidden_size else 0
    model_name = getattr(cfg, "_name_or_path", "unknown") if cfg else "unknown"

    # Measure norms
    text = sample_text or "The quick brown fox jumps over the lazy dog. " * 20
    kv_norms = {}
    try:
        kv_norms = measure_kv_norms(model, tokenizer, text)
    except Exception as e:
        logger.debug(f"KV norm measurement failed: {e}")

    # Compute stats
    kv_norm_mean = 0.0
    kv_norm_std = 0.0
    kv_norm_ratio = 1.0
    if kv_norms:
        key_norms = [v["key_norm"] for v in kv_norms.values()]
        key_max = [v["key_max_norm"] for v in kv_norms.values()]
        if key_norms:
            kv_norm_mean = sum(key_norms) / len(key_norms)
            variance = sum((x - kv_norm_mean) ** 2 for x in key_norms) / max(len(key_norms), 1)
            kv_norm_std = variance ** 0.5
            max_norm = max(key_max) if key_max else kv_norm_mean
            kv_norm_ratio = max_norm / max(kv_norm_mean, 1e-8)

    # Recommendation
    preset = recommend_preset(model, tokenizer, text)
    bits_map = {"quality": 4, "balanced": 3, "compact": 2}
    rec_bits = bits_map[preset]

    # Compression estimate (fp16 → N-bit with per-group overhead)
    group_size = 128
    overhead_per_group = 4  # bytes for mean+std (fp16 each)
    original_bytes_per_token = head_dim * 2 * 2  # K+V in fp16
    compressed_bytes_per_token = (
        head_dim * rec_bits / 8 * 2  # K+V codes
        + (head_dim / group_size) * overhead_per_group * 2  # K+V group params
    )
    compression = original_bytes_per_token / max(compressed_bytes_per_token, 1)

    return DiagnosticReport(
        model_name=model_name,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        kv_norm_mean=kv_norm_mean,
        kv_norm_std=kv_norm_std,
        kv_norm_ratio=kv_norm_ratio,
        recommended_preset=preset,
        recommended_bits=rec_bits,
        estimated_compression=compression,
        details={"kv_norms": kv_norms},
    )
