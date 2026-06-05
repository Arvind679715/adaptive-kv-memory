"""Test how outlier channels in K destroy 3-bit Hadamard quantization.

Real LLM K has channel-wise outliers: a few dims have ~100-1000x larger
magnitude than the rest. Hadamard rotation is supposed to smooth these out,
but if outliers are extreme it isn't enough.
"""
import torch
from akv.turbo_quant import TurboQuantizer, TurboQuantConfig

torch.manual_seed(0)

H, D, N = 2, 128, 1024


def per_token_cos(orig, recon):
    o = orig.reshape(-1, D).float()
    r = recon.reshape(-1, D).float()
    n = (o * r).sum(-1) / (o.norm(dim=-1) * r.norm(dim=-1) + 1e-6)
    return n


def k_with_outliers(outlier_scale: float, n_outliers: int) -> torch.Tensor:
    """Build K (H, N, D) with per-channel outliers."""
    k = torch.randn(H, N, D, dtype=torch.float32) * 0.5
    # Pick a few channels to be outliers (same across all positions)
    outlier_dims = torch.randperm(D)[:n_outliers]
    # Outlier scale: channel-wise multiplier
    scale = torch.ones(D)
    scale[outlier_dims] = outlier_scale
    k = k * scale  # broadcast over (H, N, D)
    # Apply RoPE
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
    pos = torch.arange(N, dtype=torch.float32)
    freqs = torch.einsum("n,d->nd", pos, inv_freq)
    cos = torch.cos(freqs); sin = torch.sin(freqs)
    half = D // 2
    x1, x2 = k[..., :half], k[..., half:]
    rot = torch.empty_like(k)
    rot[..., :half] = x1 * cos - x2 * sin
    rot[..., half:] = x1 * sin + x2 * cos
    return rot.to(torch.float16)


for outlier_scale, n_out, label in [
    (1.0,   0,  "no outliers           "),
    (10.0,  3,  "3 ch @ 10x outlier    "),
    (50.0,  3,  "3 ch @ 50x outlier    "),
    (100.0, 3,  "3 ch @ 100x outlier   "),
    (500.0, 3,  "3 ch @ 500x outlier   "),
    (100.0, 10, "10 ch @ 100x outlier  "),
]:
    print(f"\n=== {label} ===")
    k = k_with_outliers(outlier_scale, n_out)
    for bits in [3, 4, 8]:
        qz = TurboQuantizer(TurboQuantConfig(
            key_bits=bits, value_bits=bits,
            group_size=64, rotation="hadamard",
        ))
        qk = qz.quantize_keys(k)
        k_recon = qz.dequantize_keys(qk)
        cos = per_token_cos(k, k_recon)
        print(f"   {bits}-bit:  cos mean={cos.mean():.4f}  "
              f"min={cos.min():.4f}  "
              f"frac<0.9={(cos < 0.9).float().mean():.3f}  "
              f"frac<0.5={(cos < 0.5).float().mean():.3f}")
