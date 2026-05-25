"""KV cache quantization: per-channel asymmetric quantization with group support.

Supports 2-bit, 4-bit, and 8-bit compression of key/value tensors.
Uses per-group asymmetric quantization for accuracy preservation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class QuantBits(IntEnum):
    INT2 = 2
    INT4 = 4
    INT8 = 8


@dataclass
class QuantConfig:
    bits: int = 4
    group_size: int = 128
    symmetric: bool = False

    def __post_init__(self):
        if self.bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")


@dataclass
class QuantizedTensor:
    """Holds a quantized tensor with its dequantization parameters."""
    data: torch.Tensor        # uint8-packed quantized values
    scales: torch.Tensor      # per-group scale factors
    zeros: torch.Tensor       # per-group zero points
    shape: tuple              # original tensor shape
    bits: int                 # quantization bits
    group_size: int           # group size used

    @property
    def nbytes(self) -> int:
        return self.data.nbytes + self.scales.nbytes + self.zeros.nbytes

    @property
    def compression_ratio(self) -> float:
        original_bytes = 1
        for s in self.shape:
            original_bytes *= s
        original_bytes *= 2  # fp16 = 2 bytes
        return original_bytes / max(self.nbytes, 1)


class KVQuantizer:
    """Quantizes and dequantizes KV cache tensors.

    Supports per-group asymmetric quantization at 2/4/8-bit precision.
    Key insight: different layers and heads have different sensitivity —
    the adaptive cache manager can choose different bit-widths per entry.
    """

    def __init__(self, config: Optional[QuantConfig] = None):
        self.config = config or QuantConfig()

    def quantize(
        self,
        tensor: torch.Tensor,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> QuantizedTensor:
        """Quantize a float tensor to n-bit representation.

        Args:
            tensor: Input tensor of shape (..., seq_len, head_dim)
            bits: Override bit-width (default: config.bits)
            group_size: Override group size (default: config.group_size)

        Returns:
            QuantizedTensor with packed data and dequantization params
        """
        bits = bits or self.config.bits
        group_size = group_size or self.config.group_size

        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        # Flatten to 2D: (num_rows, features)
        flat = tensor.reshape(-1, tensor.shape[-1]).float()
        num_rows, features = flat.shape

        # Pad features to multiple of group_size
        if features % group_size != 0:
            pad_size = group_size - (features % group_size)
            flat = torch.nn.functional.pad(flat, (0, pad_size))
            features = flat.shape[1]

        num_groups = features // group_size
        grouped = flat.reshape(num_rows, num_groups, group_size)

        # Compute per-group min/max
        g_min = grouped.amin(dim=-1, keepdim=True)
        g_max = grouped.amax(dim=-1, keepdim=True)

        max_val = (1 << bits) - 1

        if self.config.symmetric:
            # Symmetric: scale around zero
            abs_max = torch.max(g_min.abs(), g_max.abs())
            scales = abs_max / (max_val / 2)
            scales = scales.clamp(min=1e-10)
            zeros = torch.zeros_like(scales)
            quantized = torch.round(grouped / scales).clamp(-max_val // 2, max_val // 2)
            quantized = (quantized + max_val // 2).to(torch.uint8)
        else:
            # Asymmetric: map [min, max] -> [0, 2^bits - 1]
            scales = (g_max - g_min) / max_val
            scales = scales.clamp(min=1e-10)
            zeros = g_min
            quantized = torch.round((grouped - zeros) / scales).clamp(0, max_val).to(torch.uint8)

        # Pack into uint8 for sub-byte storage
        packed = self._pack(quantized.reshape(num_rows, -1), bits)

        # Squeeze group dim from scales/zeros: (num_rows, num_groups)
        scales = scales.squeeze(-1).to(dtype).to(device)
        zeros = zeros.squeeze(-1).to(dtype).to(device)
        packed = packed.to(device)

        return QuantizedTensor(
            data=packed,
            scales=scales,
            zeros=zeros,
            shape=original_shape,
            bits=bits,
            group_size=group_size,
        )

    def dequantize(self, qtensor: QuantizedTensor) -> torch.Tensor:
        """Dequantize back to float tensor.

        Args:
            qtensor: Quantized tensor from quantize()

        Returns:
            Reconstructed float tensor matching original shape
        """
        bits = qtensor.bits
        group_size = qtensor.group_size
        original_shape = qtensor.shape
        device = qtensor.scales.device
        dtype = qtensor.scales.dtype

        # Compute expected dimensions
        flat_rows = 1
        for s in original_shape[:-1]:
            flat_rows *= s
        features = original_shape[-1]

        # Pad features to multiple of group_size (same as quantize)
        padded_features = features
        if features % group_size != 0:
            padded_features = features + (group_size - features % group_size)

        # Unpack
        unpacked = self._unpack(qtensor.data, bits, flat_rows * padded_features)
        unpacked = unpacked.reshape(flat_rows, -1, group_size).float()

        # Expand scales/zeros to match group structure
        scales = qtensor.scales.float().unsqueeze(-1)   # (rows, groups, 1)
        zeros = qtensor.zeros.float().unsqueeze(-1)

        # Check if symmetric (zeros are all zero)
        is_symmetric = (zeros == 0).all()
        if is_symmetric:
            # Symmetric dequant: undo the offset added during quantize
            max_val = (1 << bits) - 1
            dequantized = (unpacked - max_val // 2) * scales
        else:
            # Asymmetric dequant
            dequantized = unpacked * scales + zeros

        # Reshape and trim padding
        dequantized = dequantized.reshape(flat_rows, -1)[:, :features]
        result = dequantized.reshape(original_shape).to(dtype).to(device)
        return result

    def _pack(self, data: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack sub-byte quantized values into uint8 tensor."""
        if bits == 8:
            return data.to(torch.uint8)

        data = data.to(torch.uint8)
        flat = data.reshape(-1)

        if bits == 4:
            # Pack two 4-bit values into one uint8
            if flat.numel() % 2 != 0:
                flat = torch.nn.functional.pad(flat, (0, 1))
            high = flat[0::2] << 4
            low = flat[1::2] & 0x0F
            packed = (high | low).to(torch.uint8)
            return packed

        if bits == 2:
            # Pack four 2-bit values into one uint8
            pad_to = (4 - flat.numel() % 4) % 4
            if pad_to > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_to))
            a = (flat[0::4] & 0x03) << 6
            b = (flat[1::4] & 0x03) << 4
            c = (flat[2::4] & 0x03) << 2
            d = flat[3::4] & 0x03
            packed = (a | b | c | d).to(torch.uint8)
            return packed

        raise ValueError(f"Unsupported bits: {bits}")

    def _unpack(self, packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
        """Unpack uint8-packed tensor back to individual values."""
        packed = packed.reshape(-1)

        if bits == 8:
            return packed[:num_elements].float()

        if bits == 4:
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            # Interleave
            unpacked = torch.stack([high, low], dim=-1).reshape(-1)
            return unpacked[:num_elements].float()

        if bits == 2:
            a = (packed >> 6) & 0x03
            b = (packed >> 4) & 0x03
            c = (packed >> 2) & 0x03
            d = packed & 0x03
            unpacked = torch.stack([a, b, c, d], dim=-1).reshape(-1)
            return unpacked[:num_elements].float()

        raise ValueError(f"Unsupported bits: {bits}")

    def estimate_error(self, tensor: torch.Tensor, bits: Optional[int] = None) -> float:
        """Estimate quantization error (MSE) for a given tensor and bit-width."""
        qtensor = self.quantize(tensor, bits=bits)
        reconstructed = self.dequantize(qtensor)
        mse = (tensor.float() - reconstructed.float()).pow(2).mean().item()
        return mse

    def estimate_memory_savings(
        self, num_layers: int, num_heads: int, seq_len: int, head_dim: int, bits: Optional[int] = None
    ) -> dict:
        """Estimate memory savings from quantization."""
        bits = bits or self.config.bits
        original_bytes = num_layers * 2 * num_heads * seq_len * head_dim * 2  # KV, fp16
        # Quantized: data + scales + zeros overhead
        data_bytes = num_layers * 2 * num_heads * seq_len * head_dim * bits / 8
        num_groups = (head_dim + self.config.group_size - 1) // self.config.group_size
        meta_bytes = num_layers * 2 * num_heads * seq_len * num_groups * 2 * 2  # scales+zeros, fp16
        total_quant = data_bytes + meta_bytes
        return {
            "original_mb": original_bytes / 1e6,
            "quantized_mb": total_quant / 1e6,
            "savings_mb": (original_bytes - total_quant) / 1e6,
            "compression_ratio": original_bytes / max(total_quant, 1),
        }
