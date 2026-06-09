"""Block-affine (min/max) quantizer for KV cache warm tier.

Implements the KIVI-style per-channel (keys) and per-token (values)
asymmetric quantization that preserves channel structure critical for
attention quality at low bit-widths (2-4 bit).

Unlike Hadamard-rotation quantizers (TurboQuant), this approach never
mixes channels, so the quantization noise stays aligned with the
natural variance structure of K/V tensors.

Interface matches TurboQuantizer so it's a drop-in replacement.

When ``packed=True`` (default), codes are stored bit-packed via
``akv.bitpack``, reducing memory by 2-4x vs uint8.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from akv.bitpack import pack, unpack


@dataclass
class AffineQuantConfig:
    """Configuration for the block-affine quantizer."""
    key_bits: int = 3
    value_bits: int = 3
    group_size: int = 64  # For grouped V quantization (0 = no grouping)
    packed: bool = True    # Bit-pack codes for memory savings
    outlier_fraction: float = 0.0  # Fraction of channels to keep at FP16 (0 = disabled)


class AffineQuantizer:
    """Block-affine asymmetric quantizer.

    Keys: per-channel quantization (one scale/zero per channel dim).
    Values: per-token quantization (one scale/zero per token position).

    When ``outlier_fraction > 0``, the top-k highest-variance channels
    (by L2 norm along the token axis) are stored at FP16 and excluded
    from quantization. This eliminates the long-tail quantization errors
    that dominate PPL at 2-3 bits.

    Returns dict with 'codes', 'scale', 'zero' compatible with
    packed_layout.measure_packed_bytes. When ``packed=True``, codes are
    stored in their bit-packed form and unpacked on dequantize.
    """

    def __init__(self, config: AffineQuantConfig):
        self.key_bits = config.key_bits
        self.value_bits = config.value_bits
        self.group_size = config.group_size
        self.packed = config.packed
        self.outlier_fraction = config.outlier_fraction

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

        # Outlier channel protection: keep top-k channels at FP16
        outlier_mask = None
        outlier_data = None
        if self.outlier_fraction > 0:
            D = k.shape[-1]
            n_outliers = max(1, int(D * self.outlier_fraction))
            # Score channels by L2 norm across tokens (high-variance = outlier)
            channel_norms = k.float().norm(dim=1).mean(dim=0)  # (D,)
            _, outlier_idx = channel_norms.topk(n_outliers)
            outlier_mask = outlier_idx
            outlier_data = k[..., outlier_idx].to(torch.float16)  # (H, N, n_outliers)
            # Zero out outlier channels before quantizing
            k = k.clone()
            k[..., outlier_idx] = 0.0

        # min/max along token axis (axis=1), keep dims for broadcast
        k_min = k.amin(dim=1, keepdim=True)  # (H, 1, D)
        k_max = k.amax(dim=1, keepdim=True)  # (H, 1, D)

        scale = (k_max - k_min) / maxval
        scale = scale.clamp(min=1e-8)  # avoid div-by-zero
        zero = k_min

        codes = ((k - zero) / scale).round().clamp(0, maxval).to(torch.uint8)

        result = {
            'scale': scale.to(torch.float16),
            'zero': zero.to(torch.float16),
            'orig_dim': codes.shape[-1],
            'bits': bits,
        }
        if outlier_mask is not None:
            result['outlier_idx'] = outlier_mask
            result['outlier_data'] = outlier_data
        if self.packed:
            result['codes'] = pack(codes, bits)
        else:
            result['codes'] = codes
        return result

    def dequantize_keys(self, qk: dict) -> torch.Tensor:
        """Dequantize keys back to fp16."""
        codes = qk['codes']
        if self.packed and qk.get('bits', self.key_bits) in (2, 3, 4):
            codes = unpack(codes, qk['bits'], qk['orig_dim'])
        codes = codes.to(torch.float16)
        scale = qk['scale']
        zero = qk['zero']
        out = codes * scale + zero
        # Restore outlier channels
        if 'outlier_idx' in qk:
            out[..., qk['outlier_idx']] = qk['outlier_data']
        return out

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

        # Outlier channel protection for values too
        outlier_mask = None
        outlier_data = None
        if self.outlier_fraction > 0:
            D = v.shape[-1]
            n_outliers = max(1, int(D * self.outlier_fraction))
            channel_norms = v.float().norm(dim=1).mean(dim=0)  # (D,)
            _, outlier_idx = channel_norms.topk(n_outliers)
            outlier_mask = outlier_idx
            outlier_data = v[..., outlier_idx].to(torch.float16)
            v = v.clone()
            v[..., outlier_idx] = 0.0

        # min/max along head_dim axis (axis=2)
        v_min = v.amin(dim=2, keepdim=True)  # (H, N, 1)
        v_max = v.amax(dim=2, keepdim=True)  # (H, N, 1)

        scale = (v_max - v_min) / maxval
        scale = scale.clamp(min=1e-8)
        zero = v_min

        codes = ((v - zero) / scale).round().clamp(0, maxval).to(torch.uint8)

        result = {
            'scale': scale.to(torch.float16),
            'zero': zero.to(torch.float16),
            'orig_dim': codes.shape[-1],
            'bits': bits,
        }
        if outlier_mask is not None:
            result['outlier_idx'] = outlier_mask
            result['outlier_data'] = outlier_data
        if self.packed:
            result['codes'] = pack(codes, bits)
        else:
            result['codes'] = codes
        return result

    def dequantize_values(self, qv: dict) -> torch.Tensor:
        """Dequantize values back to fp16."""
        codes = qv['codes']
        if self.packed and qv.get('bits', self.value_bits) in (2, 3, 4):
            codes = unpack(codes, qv['bits'], qv['orig_dim'])
        codes = codes.to(torch.float16)
        scale = qv['scale']
        zero = qv['zero']
        out = codes * scale + zero
        if 'outlier_idx' in qv:
            out[..., qv['outlier_idx']] = qv['outlier_data']
        return out
