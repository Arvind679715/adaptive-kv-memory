"""Test how badly TurboQuantizer corrupts RoPE-rotated K vs raw K.

Hypothesis: 3-bit Hadamard quantization of RoPE-rotated K makes all K
vectors collapse to roughly the same direction (cosine sim ~= 0.86 in
the Kaggle proxy diagnostic), which would make attention(Q,K) ~= uniform,
giving the catastrophic 17K PPL.
"""
import math
import torch
from akv.turbo_quant import TurboQuantizer, TurboQuantConfig

torch.manual_seed(0)

# Qwen-1.5B kv-head shape: (B=1, H_kv=2, N, D=128)
H, D = 2, 128


def build_rope_k(N: int) -> torch.Tensor:
    """Synthesize RoPE-rotated K like what Qwen passes to update()."""
    # Raw K: standard transformer activation magnitude, fp16-friendly
    raw = torch.randn(1, H, N, D, dtype=torch.float32) * 0.5
    # Apply RoPE: split D into pairs, rotate by per-position angle
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))  # (D/2,)
    pos = torch.arange(N, dtype=torch.float32)  # (N,)
    freqs = torch.einsum("n,d->nd", pos, inv_freq)  # (N, D/2)
    cos = torch.cos(freqs)  # (N, D/2)
    sin = torch.sin(freqs)
    # Rotate the first/second half
    half = D // 2
    x1, x2 = raw[..., :half], raw[..., half:]
    rotated = torch.empty_like(raw)
    rotated[..., :half] = x1 * cos - x2 * sin
    rotated[..., half:] = x1 * sin + x2 * cos
    return rotated.to(torch.float16)


def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) -> (N, N) cosine similarities."""
    xn = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    return xn @ xn.T


def report(name: str, k_orig: torch.Tensor, k_recon: torch.Tensor):
    # Flatten heads: (H, N, D) -> (H*N, D)
    o = k_orig.squeeze(0).reshape(-1, D).float()
    r = k_recon.squeeze(0).reshape(-1, D).float()
    # Per-token cosine sim between original and reconstruction
    per_tok_cos = (
        (o * r).sum(-1) / (o.norm(dim=-1) * r.norm(dim=-1) + 1e-6)
    ).mean().item()
    # Pairwise cosine sim WITHIN reconstructed K
    csm = cosine_sim_matrix(r)
    csm.fill_diagonal_(0.0)
    pair_mean = csm.abs().mean().item()
    pair_max = csm.abs().max().item()
    # MSE
    mse = ((o - r) ** 2).mean().item()
    print(
        f"  {name:30s}  "
        f"recon_cos={per_tok_cos:.4f}  "
        f"pair|cos|_mean={pair_mean:.4f}  "
        f"pair|cos|_max={pair_max:.4f}  "
        f"mse={mse:.4f}"
    )


for bits in [8, 4, 3, 2]:
    print(f"\n=== {bits}-bit ===")
    qz = TurboQuantizer(TurboQuantConfig(
        key_bits=bits, value_bits=bits,
        group_size=128, rotation="hadamard",
    ))
    for N in [256, 1024]:
        k_rope = build_rope_k(N)
        # quantize_keys takes (H, N, D)
        qk = qz.quantize_keys(k_rope.squeeze(0))
        k_recon = qz.dequantize_keys(qk).unsqueeze(0)
        report(f"RoPE K  N={N:4d}", k_rope, k_recon)

# For comparison, raw (non-RoPE) K
print("\n=== reference: raw (non-RoPE) K at 3-bit ===")
qz3 = TurboQuantizer(TurboQuantConfig(
    key_bits=3, value_bits=3,
    group_size=128, rotation="hadamard",
))
raw_k = torch.randn(1, H, 256, D, dtype=torch.float16) * 0.5
qk = qz3.quantize_keys(raw_k.squeeze(0))
k_recon = qz3.dequantize_keys(qk).unsqueeze(0)
report("raw K   N=256", raw_k, k_recon)
