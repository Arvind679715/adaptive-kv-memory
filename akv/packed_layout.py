"""Packed INT4/INT2 KV cache layout for GPU-efficient inference.

Key design decisions:
- 2 INT4 values packed per uint8 (high nibble / low nibble)
- 4 INT2 values packed per uint8
- Contiguous layout: [num_heads, seq_len, head_dim // pack_factor]
- Per-head, per-group quantization (not global)
- Preallocated memory arenas — zero dynamic allocation during decode

Memory layout:
  packed_data: (num_heads, max_seq_len, head_dim // pack_factor) uint8
  scales:      (num_heads, max_seq_len, num_groups) float16
  zeros:       (num_heads, max_seq_len, num_groups) float16

This layout is:
- Coalesced for attention (heads are outer dim)
- Compatible with Triton tiled access
- Zero-copy between tiers (just update metadata)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PackedKVConfig:
    """Configuration for packed KV cache layout."""
    max_seq_len: int = 8192
    num_heads: int = 32
    head_dim: int = 128
    bits: int = 4          # 2 or 4
    group_size: int = 128  # per-head group quantization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    @property
    def pack_factor(self) -> int:
        """Number of values packed per byte."""
        return 8 // self.bits

    @property
    def packed_dim(self) -> int:
        """head_dim after packing."""
        return self.head_dim // self.pack_factor

    @property
    def num_groups(self) -> int:
        """Number of quantization groups per head_dim."""
        return math.ceil(self.head_dim / self.group_size)


class PackedKVArena:
    """Preallocated memory arena for packed KV cache.

    Allocates once, never again. All operations during decode
    are index writes into the preallocated buffer.

    Usage:
        arena = PackedKVArena(config)
        arena.append_quantized(pos=100, heads_data=packed, scales=s, zeros=z)
        # During attention: arena.get_slice(start, end) -> packed data
    """

    def __init__(self, config: PackedKVConfig):
        self.config = config
        self._len = 0  # Current number of stored tokens

        device = config.device
        H, S, D_packed = config.num_heads, config.max_seq_len, config.packed_dim
        G = config.num_groups

        # Preallocate contiguous buffers — NEVER reallocated
        self.data = torch.zeros(H, S, D_packed, dtype=torch.uint8, device=device)
        self.scales = torch.zeros(H, S, G, dtype=config.dtype, device=device)
        self.zeros = torch.zeros(H, S, G, dtype=config.dtype, device=device)

        # Position tracking (maps arena slot → original sequence position)
        self.positions = torch.zeros(S, dtype=torch.int32, device=device)

        # Valid mask (for sparse access patterns)
        self.valid = torch.zeros(S, dtype=torch.bool, device=device)

    @property
    def length(self) -> int:
        return self._len

    @property
    def capacity(self) -> int:
        return self.config.max_seq_len

    @property
    def bytes_used(self) -> int:
        return (self.data[:, :self._len].nbytes +
                self.scales[:, :self._len].nbytes +
                self.zeros[:, :self._len].nbytes)

    def reset(self):
        """Reset arena without deallocation."""
        self._len = 0
        self.valid[:] = False

    def quantize_and_append(
        self,
        keys: torch.Tensor,  # (num_heads, N, head_dim) fp16
    ) -> int:
        """Quantize fp16 keys and append to arena in-place.

        Returns the starting position in the arena.

        This is the hot path during tier demotion. Uses vectorized
        per-head per-group quantization with no intermediate allocation.
        """
        H, N, D = keys.shape
        cfg = self.config
        start = self._len

        if start + N > cfg.max_seq_len:
            # Arena full — evict oldest
            evict_n = N
            self._evict_oldest(evict_n)
            start = self._len

        # Per-head, per-group quantization (vectorized)
        # Reshape to (H, N, num_groups, group_size)
        padded_D = cfg.num_groups * cfg.group_size
        if D < padded_D:
            keys_padded = torch.nn.functional.pad(keys, (0, padded_D - D))
        else:
            keys_padded = keys

        grouped = keys_padded.reshape(H, N, cfg.num_groups, cfg.group_size)

        # Per-group min/max → asymmetric quantization
        g_min = grouped.amin(dim=-1)  # (H, N, G)
        g_max = grouped.amax(dim=-1)  # (H, N, G)

        max_val = (1 << cfg.bits) - 1
        scales = (g_max - g_min) / max_val
        scales = scales.clamp(min=1e-10)
        zeros = g_min

        # Quantize: (grouped - zeros) / scales → [0, max_val]
        quantized = ((grouped - zeros.unsqueeze(-1)) / scales.unsqueeze(-1))
        quantized = quantized.round().clamp(0, max_val).to(torch.uint8)
        quantized = quantized.reshape(H, N, -1)[:, :, :D]  # Remove padding

        # Pack into uint8
        packed = self._pack(quantized, cfg.bits)  # (H, N, D_packed)

        # Write to arena (no allocation!)
        self.data[:, start:start + N, :] = packed
        self.scales[:, start:start + N, :] = scales.to(cfg.dtype)
        self.zeros[:, start:start + N, :] = zeros.to(cfg.dtype)
        self.valid[start:start + N] = True
        self._len = start + N

        return start

    def dequantize_slice(
        self,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Dequantize a contiguous slice. Returns (H, N, D) fp16.

        This should RARELY be called during inference — the fused
        Triton kernel handles dequant inside attention instead.
        Only use for debugging or cold-tier retrieval.
        """
        cfg = self.config
        N = end - start

        packed = self.data[:, start:end, :]       # (H, N, D_packed)
        scales = self.scales[:, start:end, :]     # (H, N, G)
        zeros = self.zeros[:, start:end, :]       # (H, N, G)

        # Unpack
        unpacked = self._unpack(packed, cfg.bits, cfg.head_dim)  # (H, N, D)

        # Dequantize per-group
        group_indices = torch.arange(cfg.head_dim, device=packed.device) // cfg.group_size
        s = scales[:, :, group_indices]  # (H, N, D)
        z = zeros[:, :, group_indices]   # (H, N, D)

        return (unpacked.float() * s.float() + z.float()).to(cfg.dtype)

    def get_packed_slice(self, start: int, end: int):
        """Get raw packed data + scales + zeros for a contiguous range.

        Used by the Triton kernel for fused dequant-attention.
        Returns views (zero-copy).
        """
        return (
            self.data[:, start:end, :],
            self.scales[:, start:end, :],
            self.zeros[:, start:end, :],
        )

    @staticmethod
    def _pack(data: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack uint8 quantized values into packed format.

        Args:
            data: (H, N, D) uint8, values in [0, 2^bits - 1]
            bits: 2 or 4

        Returns:
            (H, N, D // pack_factor) uint8
        """
        if bits == 4:
            # Pack 2 values per byte: high nibble | low nibble
            assert data.shape[-1] % 2 == 0
            high = data[..., 0::2] << 4
            low = data[..., 1::2] & 0x0F
            return (high | low).to(torch.uint8)
        elif bits == 2:
            # Pack 4 values per byte
            assert data.shape[-1] % 4 == 0
            a = (data[..., 0::4] & 0x03) << 6
            b = (data[..., 1::4] & 0x03) << 4
            c = (data[..., 2::4] & 0x03) << 2
            d = data[..., 3::4] & 0x03
            return (a | b | c | d).to(torch.uint8)
        else:
            return data

    @staticmethod
    def _unpack(packed: torch.Tensor, bits: int, target_dim: int) -> torch.Tensor:
        """Unpack packed uint8 to individual values.

        Args:
            packed: (H, N, D_packed) uint8
            bits: 2 or 4
            target_dim: original head_dim

        Returns:
            (H, N, target_dim) float32
        """
        if bits == 4:
            high = ((packed >> 4) & 0x0F).float()
            low = (packed & 0x0F).float()
            # Interleave
            H, N, D_packed = packed.shape
            result = torch.empty(H, N, D_packed * 2, device=packed.device)
            result[..., 0::2] = high
            result[..., 1::2] = low
            return result[..., :target_dim]
        elif bits == 2:
            a = ((packed >> 6) & 0x03).float()
            b = ((packed >> 4) & 0x03).float()
            c = ((packed >> 2) & 0x03).float()
            d = (packed & 0x03).float()
            H, N, D_packed = packed.shape
            result = torch.empty(H, N, D_packed * 4, device=packed.device)
            result[..., 0::4] = a
            result[..., 1::4] = b
            result[..., 2::4] = c
            result[..., 3::4] = d
            return result[..., :target_dim]
        else:
            return packed.float()

    def _evict_oldest(self, n: int):
        """Evict n oldest entries by shifting data. In production,
        use a circular buffer instead."""
        if n >= self._len:
            self.reset()
            return
        remaining = self._len - n
        self.data[:, :remaining] = self.data[:, n:self._len].clone()
        self.scales[:, :remaining] = self.scales[:, n:self._len].clone()
        self.zeros[:, :remaining] = self.zeros[:, n:self._len].clone()
        self.positions[:remaining] = self.positions[n:self._len].clone()
        self.valid[:remaining] = True
        self.valid[remaining:self._len] = False
        self._len = remaining


class PagedKVCache:
    """Paged KV cache with fixed-size blocks.

    Inspired by vLLM's PagedAttention. Instead of monolithic tensors,
    KV pairs are stored in fixed-size pages. This eliminates:
    - Memory fragmentation
    - Dynamic reallocation
    - torch.cat() during decode

    Page layout:
        page_size tokens per page
        Pages stored contiguously in a page pool
        Page table maps (layer, seq_position) → page_id
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,     # tokens per page
        max_pages: int = 4096,   # total page pool size
        dtype: torch.dtype = torch.float16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.device = device

        # Preallocate page pool: all KV data lives here
        # K pages: (max_pages, num_heads, page_size, head_dim)
        # V pages: same
        self.k_pool = torch.zeros(
            max_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
        self.v_pool = torch.zeros(
            max_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )

        # Page allocation tracking
        self._free_pages = list(range(max_pages))
        self._used_pages: set = set()

        # Per-layer page tables: maps logical sequence position to (page_id, offset)
        # page_table[layer][seq_block_idx] = page_id
        self._page_tables: list[list[int]] = [[] for _ in range(num_layers)]

        # Per-layer current fill level in the last page
        self._page_offsets: list[int] = [0] * num_layers  # tokens in current page
        self._seq_lens: list[int] = [0] * num_layers

    def allocate_page(self) -> int:
        """Allocate a page from the pool. Returns page_id."""
        if not self._free_pages:
            raise RuntimeError("Page pool exhausted")
        page_id = self._free_pages.pop()
        self._used_pages.add(page_id)
        return page_id

    def free_page(self, page_id: int):
        """Return a page to the pool."""
        if page_id in self._used_pages:
            self._used_pages.discard(page_id)
            self._free_pages.append(page_id)
            # Zero out for safety
            self.k_pool[page_id].zero_()
            self.v_pool[page_id].zero_()

    def append(
        self,
        layer_idx: int,
        keys: torch.Tensor,    # (num_heads, N, head_dim)
        values: torch.Tensor,  # (num_heads, N, head_dim)
    ):
        """Append N tokens to a layer's cache. Zero allocation during append."""
        N = keys.shape[1]
        offset = self._page_offsets[layer_idx]
        page_table = self._page_tables[layer_idx]

        pos = 0
        while pos < N:
            # Allocate new page if needed
            if offset == 0 or offset >= self.page_size:
                page_id = self.allocate_page()
                page_table.append(page_id)
                offset = 0

            # Fill current page
            page_id = page_table[-1]
            space = self.page_size - offset
            write_n = min(space, N - pos)

            self.k_pool[page_id, :, offset:offset + write_n, :] = keys[:, pos:pos + write_n, :]
            self.v_pool[page_id, :, offset:offset + write_n, :] = values[:, pos:pos + write_n, :]

            offset += write_n
            pos += write_n

        self._page_offsets[layer_idx] = offset
        self._seq_lens[layer_idx] += N

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather all KV for a layer into contiguous tensors.

        For production, the Triton kernel should read pages directly
        via the page table. This method is for compatibility.
        """
        page_table = self._page_tables[layer_idx]
        seq_len = self._seq_lens[layer_idx]

        if seq_len == 0:
            return (torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
                    torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device))

        # Gather pages
        page_ids = torch.tensor(page_table, dtype=torch.long, device=self.device)
        k_pages = self.k_pool[page_ids]  # (num_pages, H, page_size, D)
        v_pages = self.v_pool[page_ids]

        # Reshape to contiguous sequence
        k_flat = k_pages.permute(1, 0, 2, 3).reshape(self.num_heads, -1, self.head_dim)
        v_flat = v_pages.permute(1, 0, 2, 3).reshape(self.num_heads, -1, self.head_dim)

        # Trim to actual length
        return k_flat[:, :seq_len, :], v_flat[:, :seq_len, :]

    def get_page_table(self, layer_idx: int) -> torch.Tensor:
        """Get page table as tensor for Triton kernel."""
        return torch.tensor(
            self._page_tables[layer_idx],
            dtype=torch.int32, device=self.device,
        )

    def get_seq_length(self, layer_idx: int) -> int:
        return self._seq_lens[layer_idx]

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def memory_usage_mb(self) -> float:
        used = len(self._used_pages)
        bytes_per_page = self.num_heads * self.page_size * self.head_dim * 2 * 2  # K+V, fp16
        return used * bytes_per_page / 1e6

    def reset(self):
        """Free all pages."""
        for page_table in self._page_tables:
            for page_id in page_table:
                self.free_page(page_id)
            page_table.clear()
        self._page_offsets = [0] * self.num_layers
        self._seq_lens = [0] * self.num_layers


# =============================================================================
# Bit-packing helpers
#
# These take a uint8 tensor of quantization indices (values in [0, 2**bits))
# and return a smaller uint8 tensor where multiple indices share each byte.
# Used by ``AKVLayer`` to report MEASURED (not formula-derived) warm-tier
# byte usage in ``memory_usage_bytes``.
#
# Layout note: packing operates on the last dim. When the last dim is not a
# clean multiple of the pack factor we pad on the right with zeros and
# remember the original length for unpacking.
# =============================================================================

def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8 bytes (high nibble, low nibble)."""
    if indices.shape[-1] % 2 != 0:
        pad = 2 - (indices.shape[-1] % 2)
        indices = torch.nn.functional.pad(indices, (0, pad))
    high = (indices[..., 0::2].to(torch.uint8) & 0x0F) << 4
    low = indices[..., 1::2].to(torch.uint8) & 0x0F
    return high | low


def pack_uint2(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4 2-bit indices into one byte."""
    if indices.shape[-1] % 4 != 0:
        pad = 4 - (indices.shape[-1] % 4)
        indices = torch.nn.functional.pad(indices, (0, pad))
    i = indices.to(torch.uint8) & 0x03
    return (
        (i[..., 0::4] << 6)
        | (i[..., 1::4] << 4)
        | (i[..., 2::4] << 2)
        |  i[..., 3::4]
    )


def pack_uint3(indices: torch.Tensor) -> torch.Tensor:
    """Pack 8 3-bit indices into 3 bytes (24 bits)."""
    if indices.shape[-1] % 8 != 0:
        pad = 8 - (indices.shape[-1] % 8)
        indices = torch.nn.functional.pad(indices, (0, pad))
    i = indices.to(torch.int32) & 0x07
    g = i.reshape(*i.shape[:-1], -1, 8)
    word = (
        (g[..., 0] << 21) | (g[..., 1] << 18)
        | (g[..., 2] << 15) | (g[..., 3] << 12)
        | (g[..., 4] << 9)  | (g[..., 5] << 6)
        | (g[..., 6] << 3)  |  g[..., 7]
    )
    b0 = ((word >> 16) & 0xFF).to(torch.uint8)
    b1 = ((word >> 8) & 0xFF).to(torch.uint8)
    b2 = (word & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(*i.shape[:-1], -1)


def measure_packed_bytes(qdata: dict, bits: int) -> int:
    """Return the on-disk byte size of a quantizer dict after bit-packing.

    Walks ``qdata`` produced by ``TurboQuantizer.quantize_keys`` /
    ``quantize_values`` and:

    1. Bit-packs ``codes`` (uint8 indices) into the tightest layout for
       the given bit-width (2/3/4 = packed; everything else = raw uint8).
    2. Adds the size of all auxiliary tensors (``group_mean``,
       ``group_std``) at their stored dtype (fp16 by default).

    The returned int is the number of bytes a production cache would
    actually keep in memory for this quantization event. Using this
    instead of a closed-form formula gives honest, measured compression
    numbers in ``AKVLayer.memory_usage_bytes``.
    """
    if 'codes' not in qdata:
        return 0
    codes = qdata['codes']
    if bits == 4:
        packed = pack_uint4(codes)
    elif bits == 3:
        packed = pack_uint3(codes)
    elif bits == 2:
        packed = pack_uint2(codes)
    else:
        packed = codes
    total = packed.element_size() * packed.numel()
    for key in ('group_mean', 'group_std', 'scale', 'zero'):
        t = qdata.get(key)
        if isinstance(t, torch.Tensor):
            total += t.element_size() * t.numel()
    return int(total)
