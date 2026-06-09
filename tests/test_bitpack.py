"""Tests for akv.bitpack pack/unpack round-trip correctness."""
import torch
import pytest

from akv.bitpack import (
    pack_uint2, unpack_uint2,
    pack_uint3, unpack_uint3,
    pack_uint4, unpack_uint4,
    pack, unpack,
)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_uint4_roundtrip(dim):
    codes = torch.randint(0, 16, (2, 32, dim), dtype=torch.uint8)
    packed = pack_uint4(codes)
    assert packed.shape[-1] == dim // 2
    recovered = unpack_uint4(packed, dim)
    assert torch.equal(codes, recovered)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_uint3_roundtrip(dim):
    codes = torch.randint(0, 8, (2, 32, dim), dtype=torch.uint8)
    packed = pack_uint3(codes)
    # 8 values -> 3 bytes, so packed_dim = dim * 3 / 8
    assert packed.shape[-1] == dim * 3 // 8
    recovered = unpack_uint3(packed, dim)
    assert torch.equal(codes, recovered)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_uint2_roundtrip(dim):
    codes = torch.randint(0, 4, (2, 32, dim), dtype=torch.uint8)
    packed = pack_uint2(codes)
    assert packed.shape[-1] == dim // 4
    recovered = unpack_uint2(packed, dim)
    assert torch.equal(codes, recovered)


def test_dispatch_helpers():
    codes = torch.randint(0, 8, (4, 16, 128), dtype=torch.uint8)
    for bits, maxval in [(2, 4), (3, 8), (4, 16)]:
        c = codes.clamp(0, maxval - 1)
        packed = pack(c, bits)
        recovered = unpack(packed, bits, 128)
        assert torch.equal(c, recovered)


def test_no_pack_for_8bit():
    codes = torch.randint(0, 256, (4, 16, 128), dtype=torch.uint8)
    packed = pack(codes, 8)
    assert torch.equal(packed, codes)
    recovered = unpack(packed, 8, 128)
    assert torch.equal(recovered, codes)
