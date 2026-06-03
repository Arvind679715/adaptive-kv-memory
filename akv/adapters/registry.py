"""Adapter registry: per-model-family configuration for AKV.

Each adapter declares:
* `supported`: whether AKV can fully manage this architecture
* `protect_initial` / `protect_recent`: attention-sink + recent-window sizes
* `default_preset`: recommended preset for the family
* `sliding_window`: model uses local attention; AKV degrades to sliding-window-aware mode
* `kv_compressed`: model already uses latent KV (MLA) — AKV should no-op
* `notes`: human-readable caveats shown by `akv info` / diagnostics

To add support for a new architecture, call `register_adapter("my_model", ...)`
or open a PR adding to this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AdapterSpec:
    """Per-architecture configuration for AKV.

    Attributes:
        model_type: HuggingFace `model_type` string (e.g. "llama", "mistral").
        family: Human-readable family name.
        supported: True if AKV fully manages this arch.
        default_preset: One of "quality", "balanced", "compact".
        protect_initial: Tokens at sequence start kept at FP16 (attention sinks).
        protect_recent: Most-recent tokens kept at FP16 (recency window).
        sliding_window: If set, model uses local attention with this window;
                        AKV will avoid demoting tokens inside the window.
        kv_compressed: Model already compresses KV (e.g. DeepSeek MLA). AKV no-ops.
        per_layer_local: Some layers use local attention (Gemma-2). List of layer
                         indices, or "auto" to detect from config.
        warm_bits_override: Force a specific bit-width regardless of preset.
        notes: Free-form caveats / known issues.
    """
    model_type: str
    family: str
    supported: bool = True
    default_preset: str = "balanced"
    protect_initial: int = 4
    protect_recent: int = 32
    sliding_window: Optional[int] = None
    kv_compressed: bool = False
    per_layer_local: Optional[str] = None  # "auto" | None
    warm_bits_override: Optional[int] = None
    notes: str = ""

    def describe(self) -> str:
        status = "supported" if self.supported else "UNSUPPORTED"
        bits = f", forced {self.warm_bits_override}b" if self.warm_bits_override else ""
        sw = f", sliding_window={self.sliding_window}" if self.sliding_window else ""
        return (
            f"{self.family} ({self.model_type}): {status}, preset={self.default_preset}"
            f"{bits}{sw}\n  protect=[init={self.protect_initial}, recent={self.protect_recent}]"
            + (f"\n  notes: {self.notes}" if self.notes else "")
        )


# ============================================================================
# Built-in adapters. PRs welcome for additional architectures.
# ============================================================================

_REGISTRY: dict[str, AdapterSpec] = {}


def register_adapter(spec: AdapterSpec) -> None:
    """Register (or override) an adapter for a model_type."""
    _REGISTRY[spec.model_type] = spec


def _register_builtins() -> None:
    # ---- Llama family: reference implementation -----------------------------
    for mt, family in [
        ("llama", "Llama / Llama-2 / Llama-3"),
        ("tinyllama", "TinyLlama"),
    ]:
        register_adapter(AdapterSpec(
            model_type=mt, family=family,
            supported=True, default_preset="balanced",
            protect_initial=4, protect_recent=32,
            notes="Reference architecture. All AKV features verified.",
        ))

    # ---- Qwen2 / Qwen2.5 ----------------------------------------------------
    register_adapter(AdapterSpec(
        model_type="qwen2", family="Qwen2 / Qwen2.5",
        supported=True, default_preset="balanced",
        protect_initial=4, protect_recent=32,
        notes="Small KV-head counts (2 in 0.5B). Per-head bit allocation "
              "operates on KV-groups rather than physical heads.",
    ))

    # ---- Mistral / Mixtral: sliding-window attention ------------------------
    register_adapter(AdapterSpec(
        model_type="mistral", family="Mistral",
        supported=True, default_preset="balanced",
        protect_initial=4, protect_recent=128,
        sliding_window=4096,
        notes="Uses 4K sliding-window attention; AKV protects the full window "
              "from demotion to avoid double-eviction.",
    ))
    register_adapter(AdapterSpec(
        model_type="mixtral", family="Mixtral (MoE)",
        supported=True, default_preset="balanced",
        protect_initial=4, protect_recent=128,
        sliding_window=4096,
        notes="Same KV-cache layout as Mistral; MoE routing is orthogonal to "
              "the cache. Verified on Mixtral-8x7B.",
    ))

    # ---- Gemma / Gemma-2 ----------------------------------------------------
    register_adapter(AdapterSpec(
        model_type="gemma", family="Gemma",
        supported=True, default_preset="quality",
        protect_initial=4, protect_recent=32,
        notes="Logit softcapping handled by the model itself; cache is "
              "standard. Quality preset recommended due to small head dim.",
    ))
    register_adapter(AdapterSpec(
        model_type="gemma2", family="Gemma-2",
        supported=True, default_preset="quality",
        protect_initial=4, protect_recent=64,
        sliding_window=4096,
        per_layer_local="auto",
        notes="Alternating local/global attention. AKV applies tiering only "
              "to global layers; local layers use the model's own SWA.",
    ))

    # ---- Phi-3 / Phi-3.5 ----------------------------------------------------
    register_adapter(AdapterSpec(
        model_type="phi3", family="Phi-3 / Phi-3.5",
        supported=True, default_preset="balanced",
        protect_initial=4, protect_recent=32,
        notes="Standard MHA + rope_scaling. Long-RoPE variants supported.",
    ))

    # ---- DeepSeek-V2 MLA ----------------------------------------------------
    register_adapter(AdapterSpec(
        model_type="deepseek_v2", family="DeepSeek-V2 (MLA)",
        supported=False, default_preset="quality",
        kv_compressed=True,
        notes="Multi-head latent attention already compresses KV into a small "
              "latent vector (~10-30x smaller than vanilla MHA). AKV no-ops "
              "and falls back to the model's native cache to avoid double "
              "compression. Track issue #DV2 for future work.",
    ))

    # ---- GPT-2 / OPT / etc.: legacy, untested -------------------------------
    for mt, family in [
        ("gpt2", "GPT-2"),
        ("opt", "OPT"),
        ("bloom", "BLOOM"),
    ]:
        register_adapter(AdapterSpec(
            model_type=mt, family=family,
            supported=True, default_preset="quality",
            protect_initial=2, protect_recent=16,
            notes="Legacy architecture, lightly tested. Use quality preset.",
        ))


_register_builtins()


def get_adapter(model_type: str) -> Optional[AdapterSpec]:
    """Look up an adapter by HuggingFace `model_type` string."""
    return _REGISTRY.get(model_type)


def list_adapters() -> list[AdapterSpec]:
    """All registered adapters, sorted by family name."""
    return sorted(_REGISTRY.values(), key=lambda s: s.family)


def resolve_for_model(model) -> AdapterSpec:
    """Inspect a HuggingFace model and return its AdapterSpec.

    Falls back to a permissive Llama-like default with a warning note for
    unknown architectures. Never raises.
    """
    model_type = None
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)

    if model_type is None:
        return AdapterSpec(
            model_type="unknown", family="Unknown",
            supported=True, default_preset="quality",
            protect_initial=4, protect_recent=32,
            notes="Could not determine model_type; using conservative defaults.",
        )

    spec = get_adapter(model_type)
    if spec is not None:
        return spec

    return AdapterSpec(
        model_type=model_type, family=model_type,
        supported=True, default_preset="quality",
        protect_initial=4, protect_recent=32,
        notes=(f"No specific adapter for '{model_type}'; using Llama-like "
               "defaults. If you hit issues, please open an issue with the "
               "model architecture details."),
    )
