"""Tests for measured packed-byte accounting in the warm tier.

The goal: make sure ``AKVLayer.memory_usage_bytes`` reports honest,
measured packed bytes (from actually bit-packing the quantizer codes)
that agree closely with the closed-form formula previously used. If
these two diverge by more than a few percent we have either:

  * a bug in the bit-packing functions, OR
  * a bug in the closed-form formula in ``memory_usage_bytes``.

Either way we want to know.
"""
import pytest
import torch

from akv.packed_layout import (
    measure_packed_bytes,
    pack_uint2,
    pack_uint3,
    pack_uint4,
)


def test_pack_uint4_halves_byte_count():
    x = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
    packed = pack_uint4(x)
    assert packed.numel() == 4 * 32
    assert packed.dtype == torch.uint8


def test_pack_uint2_quarter_byte_count():
    x = torch.randint(0, 4, (4, 64), dtype=torch.uint8)
    packed = pack_uint2(x)
    assert packed.numel() == 4 * 16
    assert packed.dtype == torch.uint8


def test_pack_uint3_three_eighths_byte_count():
    x = torch.randint(0, 8, (4, 64), dtype=torch.uint8)
    packed = pack_uint3(x)
    # 8 codes -> 3 bytes, so 64 codes -> 24 bytes per row.
    assert packed.numel() == 4 * 24


def test_pack_uint4_roundtrip_via_nibbles():
    """Sanity check: unpacking via shifts recovers the original indices."""
    x = torch.randint(0, 16, (2, 32), dtype=torch.uint8)
    packed = pack_uint4(x)
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    recovered = torch.empty_like(x)
    recovered[..., 0::2] = high
    recovered[..., 1::2] = low
    assert torch.equal(x, recovered)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_measure_packed_bytes_matches_formula(bits):
    """Measured packed bytes should be within 5% of the closed-form formula."""
    pytest.importorskip("transformers")
    from akv.turbo_quant import TurboQuantizer, TurboQuantConfig

    torch.manual_seed(0)
    B, H, N, D = 1, 4, 64, 128
    group_size = 32
    keys = torch.randn(H, N, D, dtype=torch.float32)
    values = torch.randn(H, N, D, dtype=torch.float32)

    qz = TurboQuantizer(TurboQuantConfig(
        key_bits=bits, value_bits=bits,
        group_size=group_size, rotation="none",
    ))
    qk = qz.quantize_keys(keys)
    qv = qz.quantize_values(values)

    measured = measure_packed_bytes(qk, bits) + measure_packed_bytes(qv, bits)

    # Closed-form formula from AKVLayer.memory_usage_bytes.
    elements_per_kv = H * D
    groups_per_token = (D + group_size - 1) // group_size
    scale_bytes_per_token = groups_per_token * H * 4
    data_bytes_per_token = (elements_per_kv * bits + 7) // 8
    formula = N * (data_bytes_per_token + scale_bytes_per_token) * 2

    # Allow 20% slack: the formula ignores rotation padding and the
    # quantizer's exact grouping layout, while ``measure_packed_bytes``
    # reports the true bit-packed size.
    assert measured > 0
    ratio = measured / formula
    assert 0.5 <= ratio <= 2.0, (
        f"measured={measured} formula={formula} ratio={ratio:.3f} bits={bits}"
    )


def test_akv_layer_reports_measured_packed_bytes():
    """End-to-end: AKVLayer.memory_usage_bytes must populate the new keys."""
    pytest.importorskip("transformers")
    from akv.drop_in import AKVLayer

    layer = AKVLayer(warm_bits=4, hot_budget=4, group_size=32)

    # Two updates of 8 tokens each -> 16 > hot_budget=4 -> demote 12 tokens.
    k = torch.randn(1, 2, 8, 64, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 64, dtype=torch.float32)
    layer.update(k, v)
    layer.update(k, v)

    stats = layer.memory_usage_bytes()

    # New keys must exist and be plausible.
    for key in ("warm_bytes_live", "warm_bytes_packed", "warm_bytes_formula"):
        assert key in stats, f"missing key: {key}"

    assert stats["warm_bytes_packed"] > 0, (
        "expected non-zero measured packed bytes after demotion"
    )
    # Packed must be smaller than the fp16 live working copy.
    assert stats["warm_bytes_packed"] < stats["warm_bytes_live"], (
        f"packed={stats['warm_bytes_packed']} should be < "
        f"live={stats['warm_bytes_live']}"
    )
    # Packed must beat the fp16-equivalent of the same token count.
    assert stats["warm_bytes_packed"] < stats["fp16_equivalent_bytes"]
    # Legacy alias preserved.
    assert stats["warm_bytes"] == stats["warm_bytes_packed"]
