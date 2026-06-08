"""Block-affine (min/max) quantizer for KV cache warm tier.

Implements the KIVI-style per-channel (keys) and per-token (values)
asymmetric quantization that preserves channel structure critical for
attention quality at low bit-widths (2-4 bit).

Unlike Hadamard-rotation quantizers (TurboQuant), this approach never
mixes channels, so the quantization noise stays aligned with the
natural variance structure of K/V tensors.

Interface matches TurboQuantizer so it's a drop-in replacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AffineQuantConfig:
    """Configuration for the block-affine quantizer."""
    key_bits: int = 3
    value_bits: int = 3
    group_size: int = 64  # For grouped V quantization (0 = no grouping)


class AffineQuantizer:
    """Block-affine asymmetric quantizer.

    Keys: per-channel quantization (one scale/zero per channel dim).
    Values: per-token quantization (one scale/zero per token position).

    Returns dict with 'codes', 'scale', 'zero' compatible with
    packed_layout.measure_packed_bytes.
    """

    def __init__(self, config: AffineQuantConfig):
        self.key_bits = config.key_bits
        self.value_bits = config.value_bits
        self.group_size = config.group_size

    # -----------------------------------------------------------------
    # Keys: per-channel asymmetric (axis=-1, i.e. head_dim)
    # Shape in: (H, N, D) — quantize along N per each (H, D) pair
    # -----------------------------------------------------------------

    def quantize_keys(self, k: torch.Tensor) -> dict:
        """Quantize keys with per-channel (head_dim) min/max.

        Input shape: (H, N, D) where H=heads, N=tokens, D=head_dim.
        Per-channel means: one scale/zero per (H, 1, D) — shared across N.
        """
        bits = self.key_bits
        maxval = (1 << bits) - 1  # e.g. 7 for 3-bit

        # min/max along token axis (axis=1), keep dims for broadcast
        k_min = k.amin(dim=1, keepdim=True)  # (H, 1, D)
        k_max = k.amax(dim=1, keepdim=True)  # (H, 1, D)

        scale = (k_max - k_min) / maxval
        scale = scale.clamp(min=1e-8)  # avoid div-by-zero
        zero = k_min

        codes = ((k - zero) / scale).round().clamp(0, maxval).to(torch.uint8)

        return {
            'codes': codes,
            'scale': scale.to(torch.float16),
            'zero': zero.to(torch.float16),
        }

    def dequantize_keys(self, qk: dict) -> torch.Tensor:
        """Dequantize keys back to fp16."""
        codes = qk['codes'].to(torch.float16)
        scale = qk['scale']
        zero = qk['zero']
        return codes * scale + zero

    # -----------------------------------------------------------------
    # Values: per-token asymmetric (axis=1, i.e. token position)
    # Shape in: (H, N, D) — quantize along D per each (H, N) pair
    # -----------------------------------------------------------------

    def quantize_values(self, v: torch.Tensor) -> dict:
        """Quantize values with per-token (position) min/max.

        Input shape: (H, N, D) where H=heads, N=tokens, D=head_dim.
        Per-token means: one scale/zero per (H, N, 1) — shared across D.
        """
        bits = self.value_bits
        maxval = (1 << bits) - 1

        # min/max along head_dim axis (axis=2)
        v_min = v.amin(dim=2, keepdim=True)  # (H, N, 1)
        v_max = v.amax(dim=2, keepdim=True)  # (H, N, 1)

        scale = (v_max - v_min) / maxval
        scale = scale.clamp(min=1e-8)
        zero = v_min

        codes = ((v - zero) / scale).round().clamp(0, maxval).to(torch.uint8)

        return {
            'codes': codes,
            'scale': scale.to(torch.float16),
            'zero': zero.to(torch.float16),
        }

    def dequantize_values(self, qv: dict) -> torch.Tensor:
        """Dequantize values back to fp16."""
        codes = qv['codes'].to(torch.float16)
        scale = qv['scale']
        zero = qv['zero']
        return codes * scale + zero
