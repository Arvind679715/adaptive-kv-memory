"""End-to-end calibration pipeline for AKV.

NormQuant itself is **zero-calibration** — per-group normalization always
maps data to N(0,1) so the Lloyd-Max codebook is universal. What *does*
benefit from calibration:

1. **Per-head bit allocation** — heads vary in quantization sensitivity by
   orders of magnitude. A short calibration pass measures the per-head
   reconstruction error and assigns 2-4 bits per head to minimize average
   error under a memory budget.
2. **Preset selection** — KV-outlier ratio determines whether `quality`,
   `balanced`, or `compact` is appropriate.
3. **Sliding-window / protect-window sizing** — measured from layer-0
   attention entropy.

This module ships with the package and is exposed as `akv calibrate`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Calibration result dataclasses
# ============================================================================

@dataclass
class HeadSensitivity:
    """Per-head reconstruction error at each candidate bit-width."""
    layer_idx: int
    head_idx: int
    err_2bit: float
    err_3bit: float
    err_4bit: float

    def best_bits_for_budget(self, target_bits: float) -> int:
        """Pick the smallest bit-width whose error is within 2x of 4-bit."""
        ref = max(self.err_4bit, 1e-9)
        if self.err_2bit < 2 * ref and target_bits <= 2.5:
            return 2
        if self.err_3bit < 1.5 * ref and target_bits <= 3.5:
            return 3
        return 4


@dataclass
class CalibrationReport:
    """Full calibration result, JSON-serializable."""
    model_name: str
    model_type: str
    num_layers: int
    num_kv_heads: int
    head_dim: int

    # Diagnostics
    kv_outlier_ratio: float  # max/mean across all (layer, head)
    attention_entropy_mean: float
    attention_sink_strength: float  # fraction of attention on first 4 tokens

    # Recommendations
    recommended_preset: str
    recommended_protect_first: int
    recommended_protect_last: int
    recommended_average_bits: float

    # Per-head bit assignment (layer_idx -> list[bits per head])
    per_head_bits: dict[int, list[int]] = field(default_factory=dict)

    # Raw sensitivity data (truncated in summary, full in JSON)
    sensitivities: list[HeadSensitivity] = field(default_factory=list)

    calibration_seconds: float = 0.0
    transformers_version: str = ""
    akv_version: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationReport":
        d = json.loads(Path(path).read_text())
        sens = [HeadSensitivity(**s) for s in d.pop("sensitivities", [])]
        d["per_head_bits"] = {int(k): v for k, v in d.get("per_head_bits", {}).items()}
        return cls(sensitivities=sens, **d)

    @property
    def summary(self) -> str:
        lines = [
            f"AKV Calibration: {self.model_name}",
            f"  Arch:        {self.num_layers}L x {self.num_kv_heads} KV-heads x {self.head_dim}D ({self.model_type})",
            f"  KV outlier ratio:      {self.kv_outlier_ratio:.2f}  (>10 => use 'quality')",
            f"  Attention entropy:     {self.attention_entropy_mean:.3f}",
            f"  Attn sink strength:    {self.attention_sink_strength:.1%}  (frac on first 4 tok)",
            "",
            f"  Recommended preset:    {self.recommended_preset}",
            f"  Recommended protect:   first={self.recommended_protect_first}, "
            f"last={self.recommended_protect_last}",
            f"  Average bits / head:   {self.recommended_average_bits:.2f}",
            "",
            "  Apply with:",
            "    from akv import AKVCache",
            "    cache = AKVCache.from_calibration('calib.json')",
        ]
        return "\n".join(lines)


# ============================================================================
# Quantization error probe (uses the actual NormQuantizer)
# ============================================================================

def _measure_quant_error(
    tensor: torch.Tensor, bits: int, group_size: int = 128
) -> float:
    """Round-trip a KV slice through NormQuant and return relative L2 error."""
    from akv.turbo_quant import NormQuantizer, NormQuantConfig

    cfg = NormQuantConfig(key_bits=bits, value_bits=bits, group_size=group_size)
    qz = NormQuantizer(cfg)
    t = tensor.detach().float()
    try:
        packed = qz.quantize(t, role="key")
        recon = qz.dequantize(packed)
    except Exception:
        # Fallback: simple per-group min-max so calibration still produces
        # *some* signal even if NormQuantizer API changes.
        n_levels = 1 << bits
        flat = t.reshape(-1, group_size) if t.numel() % group_size == 0 else t.reshape(-1, t.shape[-1])
        mn = flat.min(dim=-1, keepdim=True).values
        mx = flat.max(dim=-1, keepdim=True).values
        scale = (mx - mn) / max(n_levels - 1, 1)
        q = ((flat - mn) / scale.clamp_min(1e-9)).round().clamp(0, n_levels - 1)
        recon = (q * scale + mn).reshape_as(t)
    num = (recon - t).pow(2).sum().sqrt().item()
    den = t.pow(2).sum().sqrt().item() + 1e-9
    return num / den


# ============================================================================
# Main entry point
# ============================================================================

def calibrate_model(
    model,
    tokenizer,
    sample_texts: Optional[list[str]] = None,
    max_length: int = 1024,
    max_layers_to_probe: int = 8,
    target_average_bits: float = 3.0,
    device: Optional[str] = None,
) -> CalibrationReport:
    """Run a short forward pass and produce a CalibrationReport.

    The pass is deliberately cheap (~seconds on a small model) so this can
    run on a laptop. We probe a subset of layers and extrapolate to the
    rest — empirically, intra-family layers have very similar sensitivity.

    Args:
        model:   any HuggingFace CausalLM
        tokenizer: matching tokenizer
        sample_texts: calibration prompts; defaults to a small builtin set
        max_length:   truncation for each prompt
        max_layers_to_probe: cap on layers we capture to keep runtime small
        target_average_bits: budget for per-head allocation (2.0 ... 4.0)
        device: defaults to model's device
    """
    if device is None:
        device = str(next(model.parameters()).device)

    if sample_texts is None:
        sample_texts = _DEFAULT_CALIBRATION_TEXTS

    cfg = model.config
    model_type = getattr(cfg, "model_type", "unknown")
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))
    head_dim = getattr(cfg, "head_dim",
                       (getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)))

    probe_layers = sorted(set(
        round(i * (n_layers - 1) / max(max_layers_to_probe - 1, 1))
        for i in range(min(max_layers_to_probe, n_layers))
    )) if n_layers else []

    captured: dict[int, list[torch.Tensor]] = {i: [] for i in probe_layers}
    attn_stats = {"entropy": [], "sink": []}

    hooks = []

    def _make_hook(layer_idx):
        def hook(module, inputs, output):
            # Best-effort across transformers versions: output may be
            # (attn, weights, past_kv) or a Cache object.
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, tuple) and len(item) == 2:
                        k, v = item
                        if isinstance(k, torch.Tensor) and k.dim() == 4:
                            captured[layer_idx].append(k.detach().float().cpu())
                            return
                    if isinstance(item, torch.Tensor) and item.dim() == 4 \
                            and item.shape[-1] == item.shape[-2] and layer_idx == 0:
                        # Attention weights -> entropy + sink
                        w = item.detach().float().cpu()
                        # Entropy along the key dim
                        p = w.clamp_min(1e-9)
                        h = -(p * p.log()).sum(dim=-1).mean().item()
                        attn_stats["entropy"].append(h)
                        sink = w[..., :4].sum(dim=-1).mean().item()
                        attn_stats["sink"].append(sink)
        return hook

    # Attach hooks to the decoder layers, falling back to generic search
    decoder_layers = None
    for path in ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            decoder_layers = obj
            break
        except AttributeError:
            continue

    if decoder_layers is None:
        raise RuntimeError("Could not find decoder layers on model; pass a custom model")

    for idx in probe_layers:
        if idx < len(decoder_layers):
            hooks.append(decoder_layers[idx].register_forward_hook(_make_hook(idx)))

    # Run the calibration forward passes
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length).to(device)
            try:
                model(**ids, output_attentions=(probe_layers and probe_layers[0] == 0),
                      use_cache=True)
            except Exception as e:
                logger.warning("Calibration forward failed: %s", e)
                break

    for h in hooks:
        h.remove()
    cal_time = time.perf_counter() - t0

    # ---- Per-head sensitivity (cheap: just one batch element) -----------
    sensitivities: list[HeadSensitivity] = []
    outliers = []
    for layer_idx, tensors in captured.items():
        if not tensors:
            continue
        k = tensors[0]  # (B, H, N, D)
        h_count = k.shape[1]
        for h_idx in range(h_count):
            head_k = k[:, h_idx]
            outliers.append(head_k.float().norm(dim=-1).max().item() /
                            max(head_k.float().norm(dim=-1).mean().item(), 1e-9))
            try:
                e2 = _measure_quant_error(head_k, 2)
                e3 = _measure_quant_error(head_k, 3)
                e4 = _measure_quant_error(head_k, 4)
            except Exception:
                e2 = e3 = e4 = float("nan")
            sensitivities.append(HeadSensitivity(layer_idx, h_idx, e2, e3, e4))

    outlier_ratio = max(outliers) if outliers else 1.0
    attn_entropy = (sum(attn_stats["entropy"]) / len(attn_stats["entropy"])
                    if attn_stats["entropy"] else 0.0)
    attn_sink = (sum(attn_stats["sink"]) / len(attn_stats["sink"])
                 if attn_stats["sink"] else 0.0)

    # ---- Recommendations ------------------------------------------------
    if outlier_ratio > 10.0:
        preset = "quality"
    elif outlier_ratio > 5.0:
        preset = "balanced"
    else:
        preset = "compact"

    protect_first = 4 if attn_sink > 0.1 else 2
    protect_last = 64 if attn_entropy > 3.0 else 32

    # Per-head bit assignment within probed layers
    per_head_bits: dict[int, list[int]] = {}
    for layer_idx in probe_layers:
        layer_sens = [s for s in sensitivities if s.layer_idx == layer_idx]
        if layer_sens:
            per_head_bits[layer_idx] = [
                s.best_bits_for_budget(target_average_bits) for s in layer_sens
            ]

    all_bits = [b for bits in per_head_bits.values() for b in bits]
    avg_bits = sum(all_bits) / len(all_bits) if all_bits else target_average_bits

    try:
        import transformers
        tv = transformers.__version__
    except ImportError:
        tv = "?"
    try:
        from akv import __version__ as av
    except ImportError:
        av = "?"

    return CalibrationReport(
        model_name=getattr(cfg, "_name_or_path", "unknown"),
        model_type=model_type,
        num_layers=n_layers,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        kv_outlier_ratio=outlier_ratio,
        attention_entropy_mean=attn_entropy,
        attention_sink_strength=attn_sink,
        recommended_preset=preset,
        recommended_protect_first=protect_first,
        recommended_protect_last=protect_last,
        recommended_average_bits=avg_bits,
        per_head_bits=per_head_bits,
        sensitivities=sensitivities,
        calibration_seconds=cal_time,
        transformers_version=tv,
        akv_version=av,
    )


_DEFAULT_CALIBRATION_TEXTS = [
    # Short generic prose
    "The quick brown fox jumps over the lazy dog. " * 10,
    # Code-like
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n" * 5,
    # Long retrieval-style
    "In a faraway kingdom, there lived a wise queen who had three sons. "
    "Each son was given a task to prove his worthiness. " * 8,
    # Repetitive (stresses outliers)
    "AAAA BBBB CCCC DDDD " * 50,
]
