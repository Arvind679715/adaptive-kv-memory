"""Bit-packing helpers for 2/3/4-bit quantized indices.

Standalone module so the quantizer can pack without importing the full
packed_layout machinery. Same logic as packed_layout but includes unpack.
"""
from __future__ import annotations

import torch


# ── 4-bit ──────────────────────────────────────────────────────────────

def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8 bytes. Last dim must be even."""
    if indices.shape[-1] % 2 != 0:
        indices = torch.nn.functional.pad(indices, (0, 2 - indices.shape[-1] % 2))
    high = (indices[..., 0::2].to(torch.uint8) & 0x0F) << 4
    low = indices[..., 1::2].to(torch.uint8) & 0x0F
    return high | low


def unpack_uint4(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Inverse of pack_uint4."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    shape = packed.shape[:-1] + (orig_dim,)
    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = high[..., :orig_dim // 2 + orig_dim % 2]
    out[..., 1::2] = low[..., :orig_dim // 2]
    return out


# ── 3-bit ──────────────────────────────────────────────────────────────

def pack_uint3(indices: torch.Tensor) -> torch.Tensor:
    """Pack 8 3-bit indices into 3 bytes (24 bits). Last dim must be multiple of 8."""
    if indices.shape[-1] % 8 != 0:
        pad = 8 - (indices.shape[-1] % 8)
        indices = torch.nn.functional.pad(indices, (0, pad))
    i = indices.to(torch.int32) & 0x07
    g = i.reshape(*i.shape[:-1], -1, 8)
    word = (
        (g[..., 0] << 21) | (g[..., 1] << 18)
        | (g[..., 2] << 15) | (g[..., 3] << 12)
        | (g[..., 4] << 9) | (g[..., 5] << 6)
        | (g[..., 6] << 3) | g[..., 7]
    )
    b0 = ((word >> 16) & 0xFF).to(torch.uint8)
    b1 = ((word >> 8) & 0xFF).to(torch.uint8)
    b2 = (word & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(*i.shape[:-1], -1)


def unpack_uint3(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Inverse of pack_uint3."""
    k = packed.shape[-1] // 3
    p = packed.reshape(*packed.shape[:-1], k, 3).to(torch.int32)
    word = (p[..., 0] << 16) | (p[..., 1] << 8) | p[..., 2]
    out = torch.empty(*packed.shape[:-1], k, 8, dtype=torch.uint8, device=packed.device)
    out[..., 0] = ((word >> 21) & 0x7).to(torch.uint8)
    out[..., 1] = ((word >> 18) & 0x7).to(torch.uint8)
    out[..., 2] = ((word >> 15) & 0x7).to(torch.uint8)
    out[..., 3] = ((word >> 12) & 0x7).to(torch.uint8)
    out[..., 4] = ((word >> 9) & 0x7).to(torch.uint8)
    out[..., 5] = ((word >> 6) & 0x7).to(torch.uint8)
    out[..., 6] = ((word >> 3) & 0x7).to(torch.uint8)
    out[..., 7] = (word & 0x7).to(torch.uint8)
    return out.reshape(*packed.shape[:-1], k * 8)[..., :orig_dim]


# ── 2-bit ──────────────────────────────────────────────────────────────

def pack_uint2(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4 2-bit indices into one byte. Last dim must be multiple of 4."""
    if indices.shape[-1] % 4 != 0:
        pad = 4 - (indices.shape[-1] % 4)
        indices = torch.nn.functional.pad(indices, (0, pad))
    i = indices.to(torch.uint8) & 0x03
    return (i[..., 0::4] << 6) | (i[..., 1::4] << 4) | (i[..., 2::4] << 2) | i[..., 3::4]


def unpack_uint2(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Inverse of pack_uint2."""
    shape = packed.shape[:-1] + (orig_dim,)
    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::4] = (packed >> 6) & 0x03
    out[..., 1::4] = (packed >> 4) & 0x03
    out[..., 2::4] = (packed >> 2) & 0x03
    out[..., 3::4] = packed & 0x03
    return out


# ── Dispatch helpers ───────────────────────────────────────────────────

def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack indices at the given bit-width."""
    if bits == 4:
        return pack_uint4(indices)
    if bits == 3:
        return pack_uint3(indices)
    if bits == 2:
        return pack_uint2(indices)
    return indices  # 8-bit or unsupported — no packing


def unpack(packed: torch.Tensor, bits: int, orig_dim: int) -> torch.Tensor:
    """Unpack indices from the given bit-width."""
    if bits == 4:
        return unpack_uint4(packed, orig_dim)
    if bits == 3:
        return unpack_uint3(packed, orig_dim)
    if bits == 2:
        return unpack_uint2(packed, orig_dim)
    return packed


def packed_bytes(codes: torch.Tensor, bits: int) -> int:
    """Return byte count after packing codes at given bit-width."""
    packed = pack(codes, bits)
    return packed.element_size() * packed.numel()
