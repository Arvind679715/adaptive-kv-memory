"""NormQuant quantization: Hadamard rotation + per-group normalization + Lloyd-Max codebook.

Builds on ideas from TurboQuant (ICLR 2026) but adds per-group normalization
which is the key enabler for high-quality low-bit KV cache quantization:
- Random Hadamard rotation smooths outlier channels before quantization
- Per-group normalization standardizes each group to N(0,1)
- Lloyd-Max codebook provides optimal non-uniform quantization levels
- Asymmetric key/value bit allocation (3b keys, 2b values)

This integrates with AKV's adaptive tiered cache as the warm-tier quantizer,
providing significantly better quality at the same bit-width vs min-max.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class NormQuantConfig:
    """Configuration for NormQuant quantization."""
    key_bits: int = 3          # Bits for key quantization
    value_bits: int = 2        # Bits for value quantization (less sensitive)
    group_size: int = 128      # Quantization group size
    codebook_size: int = 0     # 0 = use 2^bits levels; >0 = custom codebook
    rotation: str = "hadamard" # "hadamard", "random", or "none"
    calibration_steps: int = 50  # Lloyd-Max iterations


# Backward-compatible alias
TurboQuantConfig = NormQuantConfig


def hadamard_matrix(n: int, device: torch.device = None) -> torch.Tensor:
    """Generate normalized Hadamard matrix of size n×n.

    Uses recursive construction. n must be a power of 2.
    Returns orthonormal matrix (H @ H.T = I).
    """
    if n == 1:
        H = torch.ones(1, 1, device=device)
    else:
        H_half = hadamard_matrix(n // 2, device=device)
        H = torch.cat([
            torch.cat([H_half, H_half], dim=1),
            torch.cat([H_half, -H_half], dim=1),
        ], dim=0)
    return H / (n ** 0.5)  # Normalize so H @ H.T = I


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply fast Walsh-Hadamard transform in O(n log n).

    Args:
        x: (..., D) where D is power of 2

    Returns:
        Rotated tensor of same shape, normalized.
    """
    D = x.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"Dimension must be power of 2, got {D}"

    # In-place butterfly computation
    result = x.clone()
    h = 1
    while h < D:
        # Split into pairs of stride h
        result_view = result.view(*result.shape[:-1], -1, 2 * h)
        left = result_view[..., :h].clone()
        right = result_view[..., h:].clone()
        result_view[..., :h] = left + right
        result_view[..., h:] = left - right
        h *= 2

    return result / (D ** 0.5)


def inverse_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Inverse Hadamard transform (same as forward for orthonormal Hadamard)."""
    return fast_hadamard_transform(x)


def lloyd_max_codebook(
    data: torch.Tensor,
    num_levels: int,
    max_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train Lloyd-Max optimal quantization codebook.

    Finds non-uniform quantization levels that minimize MSE for the
    given data distribution. This is the 1D k-means equivalent.

    Args:
        data: Flat tensor of values to fit
        num_levels: Number of quantization levels (2^bits)
        max_iter: Maximum iterations

    Returns:
        (levels, boundaries) where:
          levels: (num_levels,) optimal reconstruction values
          boundaries: (num_levels-1,) decision boundaries
    """
    data = data.float().flatten()

    # Initialize levels uniformly between min and max
    d_min, d_max = data.min().item(), data.max().item()
    if d_min == d_max:
        levels = torch.full((num_levels,), d_min, device=data.device)
        boundaries = torch.linspace(d_min, d_max, num_levels - 1, device=data.device)
        return levels, boundaries

    levels = torch.linspace(d_min, d_max, num_levels, device=data.device)

    for _ in range(max_iter):
        # Compute boundaries (midpoints between adjacent levels)
        boundaries = (levels[:-1] + levels[1:]) / 2.0

        # Assign each data point to nearest level
        # data: (N,), boundaries: (num_levels-1,)
        assignments = torch.bucketize(data, boundaries)  # (N,) in [0, num_levels-1]

        # Update levels to centroid of assigned points
        new_levels = torch.zeros_like(levels)
        for i in range(num_levels):
            mask = assignments == i
            if mask.any():
                new_levels[i] = data[mask].mean()
            else:
                new_levels[i] = levels[i]  # Keep if no points assigned

        # Check convergence
        if torch.allclose(levels, new_levels, atol=1e-6):
            break
        levels = new_levels

    # Final boundaries
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    return levels.sort()[0], boundaries.sort()[0]


class NormQuantizer:
    """NormQuant quantizer with rotation + per-group normalization + optimal codebook.

    Drop-in replacement for KVQuantizer with better quality at same bits.

    Usage:
        tq = NormQuantizer(NormQuantConfig(key_bits=3, value_bits=2))
        # Calibrate on representative data (once)
        tq.calibrate(sample_keys, sample_values)
        # Quantize
        q_keys = tq.quantize_keys(keys)
        q_values = tq.quantize_values(values)
        # Dequantize
        keys_recon = tq.dequantize_keys(q_keys)
    """

    def __init__(self, config: NormQuantConfig = None):
        self.config = config or NormQuantConfig()
        self._key_codebook: Optional[torch.Tensor] = None  # (num_levels,)
        self._key_boundaries: Optional[torch.Tensor] = None
        self._value_codebook: Optional[torch.Tensor] = None
        self._value_boundaries: Optional[torch.Tensor] = None
        self._rotation_seed: int = 42  # For reproducible random rotation
        self._calibrated = False

    def calibrate(
        self,
        sample_keys: torch.Tensor,
        sample_values: torch.Tensor,
    ):
        """Calibrate codebooks from representative KV data.

        Trains Lloyd-Max codebooks on PER-GROUP NORMALIZED data.
        Since each group is normalized to ~N(0,1) at quantize time,
        the codebook learns optimal levels for the standard normal distribution.

        Args:
            sample_keys: (H, N, D) or (B, H, N, D) sample keys
            sample_values: same shape, sample values
        """
        cfg = self.config

        # Rotate samples (same transform used during quantization)
        rotated_keys = self._rotate(sample_keys.float())
        rotated_values = self._rotate(sample_values.float())

        # Group for per-group normalization
        key_groups = self._group(rotated_keys)    # (num_groups, group_size)
        value_groups = self._group(rotated_values)

        # Normalize each group to zero-mean/unit-variance (matches quantize-time normalization)
        key_mean = key_groups.mean(dim=1, keepdim=True)
        key_std = key_groups.std(dim=1, keepdim=True).clamp(min=1e-8)
        key_normalized = ((key_groups - key_mean) / key_std).flatten()

        value_mean = value_groups.mean(dim=1, keepdim=True)
        value_std = value_groups.std(dim=1, keepdim=True).clamp(min=1e-8)
        value_normalized = ((value_groups - value_mean) / value_std).flatten()

        # Subsample if too large (codebook training on 100K points is sufficient)
        max_samples = 100_000
        if key_normalized.numel() > max_samples:
            idx = torch.randperm(key_normalized.numel(), device=key_normalized.device)[:max_samples]
            key_normalized = key_normalized[idx]
        if value_normalized.numel() > max_samples:
            idx = torch.randperm(value_normalized.numel(), device=value_normalized.device)[:max_samples]
            value_normalized = value_normalized[idx]

        # Train Lloyd-Max codebooks on normalized data (~standard normal)
        num_key_levels = 1 << cfg.key_bits
        num_value_levels = 1 << cfg.value_bits

        self._key_codebook, self._key_boundaries = lloyd_max_codebook(
            key_normalized, num_key_levels, cfg.calibration_steps
        )
        self._value_codebook, self._value_boundaries = lloyd_max_codebook(
            value_normalized, num_value_levels, cfg.calibration_steps
        )
        self._calibrated = True

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation to smooth outlier channels."""
        if self.config.rotation == "none":
            return x

        D = x.shape[-1]
        # Pad to power of 2 if needed
        D_padded = 1 << (D - 1).bit_length()
        if D_padded != D:
            x = F.pad(x, (0, D_padded - D))

        if self.config.rotation == "hadamard":
            return fast_hadamard_transform(x)
        elif self.config.rotation == "random":
            # Random sign flip + Hadamard (randomized Hadamard)
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self._rotation_seed)
            signs = torch.randint(0, 2, (D_padded,), generator=gen, device=x.device) * 2 - 1
            x = x * signs.float()
            return fast_hadamard_transform(x)
        else:
            return x

    def _unrotate(self, x: torch.Tensor, original_dim: int) -> torch.Tensor:
        """Inverse rotation."""
        if self.config.rotation == "none":
            return x[..., :original_dim]

        if self.config.rotation == "hadamard":
            result = inverse_hadamard_transform(x)
        elif self.config.rotation == "random":
            result = inverse_hadamard_transform(x)
            D_padded = x.shape[-1]
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self._rotation_seed)
            signs = torch.randint(0, 2, (D_padded,), generator=gen, device=x.device) * 2 - 1
            result = result * signs.float()
        else:
            result = x

        return result[..., :original_dim]

    def _group(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to groups for per-group quantization."""
        shape = x.shape
        D = shape[-1]
        gs = self.config.group_size
        if D % gs != 0:
            x = F.pad(x, (0, gs - D % gs))
        return x.reshape(-1, self.config.group_size)

    def quantize_keys(self, keys: torch.Tensor) -> dict:
        """Quantize key tensor using rotation + per-group normalization + Lloyd-Max codebook.

        Args:
            keys: (H, N, D) or (B, H, N, D) fp16/fp32 key tensor

        Returns:
            Dict with codes, per-group scales, and shape metadata.
        """
        assert self._calibrated, "Must call calibrate() first"
        original_shape = keys.shape
        original_dim = keys.shape[-1]

        rotated = self._rotate(keys.float())
        rotated_dim = rotated.shape[-1]
        grouped = self._group(rotated)  # (num_groups, group_size)

        # Per-group normalization: normalize to zero-mean/unit-variance
        group_mean = grouped.mean(dim=1, keepdim=True)  # (num_groups, 1)
        group_std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (grouped - group_mean) / group_std

        # Quantize normalized values using codebook boundaries
        codes = torch.bucketize(normalized, self._key_boundaries)

        # Compute grouped shape (accounts for padding to group_size)
        gs = self.config.group_size
        D_grouped = rotated_dim if rotated_dim % gs == 0 else rotated_dim + (gs - rotated_dim % gs)
        grouped_shape = (*rotated.shape[:-1], D_grouped)

        return {
            'codes': codes.to(torch.uint8),
            'group_mean': group_mean.squeeze(1).to(torch.float16),  # (num_groups,)
            'group_std': group_std.squeeze(1).to(torch.float16),
            'shape': original_shape,
            'original_dim': original_dim,
            'rotated_shape': rotated.shape,
            'grouped_shape': grouped_shape,
        }

    def dequantize_keys(self, qdata: dict) -> torch.Tensor:
        """Dequantize keys back to float."""
        codes = qdata['codes'].long()
        original_shape = qdata['shape']
        original_dim = qdata['original_dim']
        rotated_shape = qdata['rotated_shape']
        grouped_shape = qdata.get('grouped_shape', rotated_shape)

        # Lookup codebook values (in normalized space)
        reconstructed = self._key_codebook[codes]  # (num_groups, group_size)

        # Denormalize with per-group scale/shift
        if 'group_mean' in qdata:
            group_mean = qdata['group_mean'].float().unsqueeze(1)  # (num_groups, 1)
            group_std = qdata['group_std'].float().unsqueeze(1)
            reconstructed = reconstructed * group_std + group_mean

        # Reshape back to grouped shape (may include padding beyond rotation dim)
        reconstructed = reconstructed.reshape(grouped_shape)

        # Remove group padding to get back to rotation dimension
        rotated_dim = rotated_shape[-1]
        reconstructed = reconstructed[..., :rotated_dim]

        # Inverse rotation
        result = self._unrotate(reconstructed, original_dim)
        return result.reshape(original_shape).to(torch.float16)

    def quantize_values(self, values: torch.Tensor) -> dict:
        """Quantize value tensor with per-group normalization + codebook."""
        assert self._calibrated, "Must call calibrate() first"
        original_shape = values.shape
        original_dim = values.shape[-1]

        rotated = self._rotate(values.float())
        rotated_dim = rotated.shape[-1]
        grouped = self._group(rotated)

        # Per-group normalization
        group_mean = grouped.mean(dim=1, keepdim=True)
        group_std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (grouped - group_mean) / group_std

        # Compute grouped shape (accounts for padding to group_size)
        gs = self.config.group_size
        D_grouped = rotated_dim if rotated_dim % gs == 0 else rotated_dim + (gs - rotated_dim % gs)
        grouped_shape = (*rotated.shape[:-1], D_grouped)

        codes = torch.bucketize(normalized, self._value_boundaries)
        return {
            'codes': codes.to(torch.uint8),
            'group_mean': group_mean.squeeze(1).to(torch.float16),
            'group_std': group_std.squeeze(1).to(torch.float16),
            'shape': original_shape,
            'original_dim': original_dim,
            'rotated_shape': rotated.shape,
            'grouped_shape': grouped_shape,
        }

    def dequantize_values(self, qdata: dict) -> torch.Tensor:
        """Dequantize values back to float."""
        codes = qdata['codes'].long()
        original_shape = qdata['shape']
        original_dim = qdata['original_dim']
        rotated_shape = qdata['rotated_shape']
        grouped_shape = qdata.get('grouped_shape', rotated_shape)

        reconstructed = self._value_codebook[codes]

        # Denormalize with per-group scale/shift
        if 'group_mean' in qdata:
            group_mean = qdata['group_mean'].float().unsqueeze(1)
            group_std = qdata['group_std'].float().unsqueeze(1)
            reconstructed = reconstructed * group_std + group_mean

        reconstructed = reconstructed.reshape(grouped_shape)

        # Remove group padding to get back to rotation dimension
        rotated_dim = rotated_shape[-1]
        reconstructed = reconstructed[..., :rotated_dim]

        result = self._unrotate(reconstructed, original_dim)
        return result.reshape(original_shape).to(torch.float16)

    def quantize_and_measure(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict:
        """Quantize and return quality metrics.

        Useful for comparing against baseline min-max quantizer.
        """
        q_keys = self.quantize_keys(keys)
        q_values = self.quantize_values(values)

        recon_keys = self.dequantize_keys(q_keys)
        recon_values = self.dequantize_values(q_values)

        # MSE
        key_mse = ((keys.float() - recon_keys.float()) ** 2).mean().item()
        value_mse = ((values.float() - recon_values.float()) ** 2).mean().item()

        # Cosine similarity
        key_cos = F.cosine_similarity(
            keys.float().flatten(), recon_keys.float().flatten(), dim=0
        ).item()
        value_cos = F.cosine_similarity(
            values.float().flatten(), recon_values.float().flatten(), dim=0
        ).item()

        # Compression ratio
        original_bytes = keys.numel() * 2 + values.numel() * 2  # fp16
        compressed_bytes = (
            q_keys['codes'].numel() * self.config.key_bits / 8 +
            q_values['codes'].numel() * self.config.value_bits / 8
        )

        return {
            'key_mse': key_mse,
            'value_mse': value_mse,
            'key_cosine': key_cos,
            'value_cosine': value_cos,
            'compression_ratio': original_bytes / compressed_bytes,
            'key_bits': self.config.key_bits,
            'value_bits': self.config.value_bits,
        }


class NormQuantWarmTier:
    """Warm-tier storage using NormQuant codebooks.

    Drop-in replacement for the (PackedKVArena K, PackedKVArena V) pair
    used in ProductionLayerCache. Provides the same interface:
    - quantize_and_append(keys/values)
    - dequantize_slice(start, end)
    - length, bytes_used, reset()

    Auto-calibrates on first migration batch, then uses fixed codebook.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        group_size: int = 128,
        device: str = "cuda",
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        self._quantizer = NormQuantizer(NormQuantConfig(
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
            rotation="hadamard",
            calibration_steps=50,
        ))

        # Preallocated code storage
        # After rotation + grouping, dim may be padded to power-of-2
        D_padded = 1 << (head_dim - 1).bit_length()
        gs = group_size
        if D_padded % gs != 0:
            D_padded_groups = D_padded + (gs - D_padded % gs)
        else:
            D_padded_groups = D_padded
        codes_per_token = D_padded_groups  # One code per element after grouping
        groups_per_token = D_padded_groups // gs

        self._k_codes = torch.zeros(
            num_heads, max_seq_len, codes_per_token,
            dtype=torch.uint8, device=device,
        )
        self._v_codes = torch.zeros(
            num_heads, max_seq_len, codes_per_token,
            dtype=torch.uint8, device=device,
        )
        # Per-group normalization side info (mean/std per group)
        self._k_mean = torch.zeros(
            num_heads, max_seq_len, groups_per_token,
            dtype=torch.float16, device=device,
        )
        self._k_std = torch.zeros(
            num_heads, max_seq_len, groups_per_token,
            dtype=torch.float16, device=device,
        )
        self._v_mean = torch.zeros(
            num_heads, max_seq_len, groups_per_token,
            dtype=torch.float16, device=device,
        )
        self._v_std = torch.zeros(
            num_heads, max_seq_len, groups_per_token,
            dtype=torch.float16, device=device,
        )
        self._groups_per_token = groups_per_token
        self._len = 0
        self._calibrated = False

    @property
    def length(self) -> int:
        return self._len

    @property
    def bytes_used(self) -> int:
        if self._len == 0:
            return 0
        k_bytes = self._k_codes[:, :self._len].numel()  # uint8 = 1 byte each
        v_bytes = self._v_codes[:, :self._len].numel()
        return k_bytes + v_bytes

    def reset(self):
        self._len = 0

    def quantize_and_append_kv(
        self,
        keys: torch.Tensor,    # (H, N, D) fp16
        values: torch.Tensor,  # (H, N, D) fp16
    ) -> int:
        """Quantize keys and values together and append to storage.

        Auto-calibrates on first call using the provided data as sample.
        Stores codes AND per-group mean/std for denormalization.
        Returns starting position.
        """
        H, N, D = keys.shape
        start = self._len

        if start + N > self.max_seq_len:
            # Evict oldest tokens to make room
            shift = N
            self._k_codes[:, :self._len - shift] = self._k_codes[:, shift:self._len].clone()
            self._v_codes[:, :self._len - shift] = self._v_codes[:, shift:self._len].clone()
            self._k_mean[:, :self._len - shift] = self._k_mean[:, shift:self._len].clone()
            self._k_std[:, :self._len - shift] = self._k_std[:, shift:self._len].clone()
            self._v_mean[:, :self._len - shift] = self._v_mean[:, shift:self._len].clone()
            self._v_std[:, :self._len - shift] = self._v_std[:, shift:self._len].clone()
            self._len -= shift
            start = self._len

        # Auto-calibrate on first migration batch
        if not self._calibrated:
            self._quantizer.calibrate(keys, values)
            self._calibrated = True

        # Quantize
        q_keys = self._quantizer.quantize_keys(keys)
        q_values = self._quantizer.quantize_values(values)

        # Store codes — reshape from (num_groups, group_size) to (H, N, codes_per_token)
        k_codes = q_keys['codes']  # shape from grouping
        v_codes = q_values['codes']

        # Flatten grouped codes back to per-head-per-token
        k_flat = k_codes.reshape(H, N, -1)
        v_flat = v_codes.reshape(H, N, -1)

        codes_dim = self._k_codes.shape[2]
        self._k_codes[:, start:start + N, :k_flat.shape[2]] = k_flat
        self._v_codes[:, start:start + N, :v_flat.shape[2]] = v_flat

        # Store per-group mean/std — reshape to (H, N, groups_per_token)
        gpt = self._groups_per_token
        k_mean = q_keys['group_mean'].reshape(H, N, gpt)
        k_std = q_keys['group_std'].reshape(H, N, gpt)
        v_mean = q_values['group_mean'].reshape(H, N, gpt)
        v_std = q_values['group_std'].reshape(H, N, gpt)

        self._k_mean[:, start:start + N, :] = k_mean
        self._k_std[:, start:start + N, :] = k_std
        self._v_mean[:, start:start + N, :] = v_mean
        self._v_std[:, start:start + N, :] = v_std

        self._len = start + N
        return start

    def dequantize_slice(
        self,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a slice. Returns (keys, values) as (H, N, D) fp16."""
        H = self.num_heads
        N = end - start
        D = self.head_dim

        # Reconstruct the qdata dict format expected by TurboQuantizer
        D_padded = 1 << (D - 1).bit_length()  # rotation dim (power of 2)
        gs = self._quantizer.config.group_size
        # Grouped dim accounts for padding to group_size alignment
        if D_padded % gs != 0:
            D_padded_groups = D_padded + (gs - D_padded % gs)
        else:
            D_padded_groups = D_padded
        rotated_shape = (H, N, D_padded)
        grouped_shape = (H, N, D_padded_groups)
        original_shape = (H, N, D)

        # Get stored codes — use full codes_per_token (D_padded_groups)
        k_codes_flat = self._k_codes[:, start:end, :D_padded_groups]
        v_codes_flat = self._v_codes[:, start:end, :D_padded_groups]

        k_grouped = k_codes_flat.reshape(-1, gs)
        v_grouped = v_codes_flat.reshape(-1, gs)

        # Retrieve stored per-group mean/std
        k_mean = self._k_mean[:, start:end, :].reshape(-1)  # (H*N*groups_per_token,)
        k_std = self._k_std[:, start:end, :].reshape(-1)
        v_mean = self._v_mean[:, start:end, :].reshape(-1)
        v_std = self._v_std[:, start:end, :].reshape(-1)

        k_qdata = {
            'codes': k_grouped,
            'group_mean': k_mean,
            'group_std': k_std,
            'shape': original_shape,
            'original_dim': D,
            'rotated_shape': rotated_shape,
            'grouped_shape': grouped_shape,
        }
        v_qdata = {
            'codes': v_grouped,
            'group_mean': v_mean,
            'group_std': v_std,
            'shape': original_shape,
            'original_dim': D,
            'rotated_shape': rotated_shape,
            'grouped_shape': grouped_shape,
        }

        recon_k = self._quantizer.dequantize_keys(k_qdata)
        recon_v = self._quantizer.dequantize_values(v_qdata)
        return recon_k, recon_v


# Backward-compatible aliases
TurboQuantizer = NormQuantizer
TurboWarmTier = NormQuantWarmTier
