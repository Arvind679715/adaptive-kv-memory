"""End-to-end AKVCache round-trip on Qwen-shape inputs.

Test: feed RoPE-rotated K/V of shape (1, 2, N, 128) into AKVCache via
update(), capture the returned full K/V tensor, compare per-token to the
ground-truth K we fed in (== what DynamicCache returns).

If per-token cosine similarity > 0.95, the cache plumbing preserves K
fidelity and the 3636x Qwen PPL bug must be in transformers integration
(cache_kwargs, RoPE, etc.). If similarity is poor, the cache plumbing
itself is corrupting K.
"""
import math
import torch
from akv.drop_in import AKVCache

torch.manual_seed(0)

H_KV, D = 2, 128            # Qwen-1.5B GQA shape
PREFILL = 1024
DECODE  = 256
HOT_BUDGET = 256


def build_rope_k(N: int, offset: int = 0) -> torch.Tensor:
    """RoPE-rotated K of shape (1, H, N, D), positions [offset, offset+N)."""
    raw = torch.randn(1, H_KV, N, D, dtype=torch.float32) * 0.5
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))   # (D/2,)
    pos = torch.arange(offset, offset + N, dtype=torch.float32)       # (N,)
    freqs = torch.einsum("n,d->nd", pos, inv_freq)                    # (N, D/2)
    cos = torch.cos(freqs); sin = torch.sin(freqs)
    half = D // 2
    x1, x2 = raw[..., :half], raw[..., half:]
    rotated = torch.empty_like(raw)
    rotated[..., :half] = x1 * cos - x2 * sin
    rotated[..., half:] = x1 * sin + x2 * cos
    return rotated.to(torch.float16)


def per_token_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (1, H, N, D) -> (N,) mean-over-heads cosine sim per position."""
    a = a.squeeze(0).float()  # (H, N, D)
    b = b.squeeze(0).float()
    num = (a * b).sum(-1)                  # (H, N)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-6
    return (num / den).mean(0)             # (N,)


def run(label: str, **kwargs):
    print(f"\n=== {label} ===")
    print(f"    kwargs: {kwargs}")
    cache = AKVCache(num_hidden_layers=1, **kwargs)

    # Build ground-truth K/V we'd push through DynamicCache
    truth_k_prefill = build_rope_k(PREFILL, offset=0)
    truth_v_prefill = torch.randn(1, H_KV, PREFILL, D, dtype=torch.float16) * 0.5

    # Prefill update
    full_k, full_v = cache.update(truth_k_prefill, truth_v_prefill, layer_idx=0)
    print(f"    after prefill: full_k.shape={tuple(full_k.shape)} "
          f"hot_len={cache.layers[0]._hot_keys.shape[2] if cache.layers[0]._hot_keys is not None else 0} "
          f"warm_len={cache.layers[0]._warm_keys_fp16.shape[2] if cache.layers[0]._warm_keys_fp16 is not None else 0}")

    cos_prefill = per_token_cos(truth_k_prefill, full_k)
    print(f"    prefill per-token cos: "
          f"mean={cos_prefill.mean():.4f}  min={cos_prefill.min():.4f}  "
          f"frac<0.5={(cos_prefill < 0.5).float().mean():.3f}  "
          f"frac<0.9={(cos_prefill < 0.9).float().mean():.3f}")

    # Decode loop
    truth_k_decode_chunks = []
    truth_v_decode_chunks = []
    final_full_k = None
    for i in range(DECODE):
        kd = build_rope_k(1, offset=PREFILL + i)
        vd = torch.randn(1, H_KV, 1, D, dtype=torch.float16) * 0.5
        truth_k_decode_chunks.append(kd)
        truth_v_decode_chunks.append(vd)
        final_full_k, final_full_v = cache.update(kd, vd, layer_idx=0)

    # Build full truth view (DynamicCache equivalent)
    truth_full_k = torch.cat([truth_k_prefill] + truth_k_decode_chunks, dim=2)
    truth_full_v = torch.cat([truth_v_prefill] + truth_v_decode_chunks, dim=2)

    print(f"    after {DECODE} decode: full_k.shape={tuple(final_full_k.shape)}")
    print(f"    truth_full_k.shape={tuple(truth_full_k.shape)}")

    if final_full_k.shape != truth_full_k.shape:
        print(f"    !!! SHAPE MISMATCH — cache lost/duplicated tokens !!!")
        return

    cos_full = per_token_cos(truth_full_k, final_full_k)
    print(f"    full per-token K cos: "
          f"mean={cos_full.mean():.4f}  min={cos_full.min():.4f}  "
          f"frac<0.5={(cos_full < 0.5).float().mean():.3f}  "
          f"frac<0.9={(cos_full < 0.9).float().mean():.3f}")
    cos_full_v = per_token_cos(truth_full_v, final_full_v)
    print(f"    full per-token V cos: "
          f"mean={cos_full_v.mean():.4f}  min={cos_full_v.min():.4f}  "
          f"frac<0.5={(cos_full_v < 0.5).float().mean():.3f}")

    # Also report the value-error magnitude (does V get destroyed?)
    v_mse = ((truth_full_v.float() - final_full_v.float()) ** 2).mean().item()
    k_mse = ((truth_full_k.float() - final_full_k.float()) ** 2).mean().item()
    print(f"    full MSE: K={k_mse:.4f}  V={v_mse:.4f}")


# Configurations matching EXP19
run("A — no-promo, 3-bit warm",
    warm_bits=3, hot_budget=HOT_BUDGET,
    enable_promotion=False, enable_promotion_proxy=False)

run("D-hyp — 8-bit warm (diagnostic)",
    warm_bits=8, hot_budget=HOT_BUDGET,
    enable_promotion=False, enable_promotion_proxy=False)

run("E-hyp — no demote (hot_budget > total)",
    warm_bits=3, hot_budget=2048,
    enable_promotion=False, enable_promotion_proxy=False)
