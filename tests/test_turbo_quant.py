"""Unit tests for TurboQuant quantizer and TurboWarmTier."""
import pytest
import torch

from akv.turbo_quant import (
    TurboQuantizer,
    TurboQuantConfig,
    TurboWarmTier,
    fast_hadamard_transform,
    inverse_hadamard_transform,
    lloyd_max_codebook,
)


class TestHadamardTransform:
    def test_roundtrip(self):
        x = torch.randn(4, 32, 64)
        y = fast_hadamard_transform(x)
        x_recon = inverse_hadamard_transform(y)
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_orthogonality(self):
        """Hadamard should preserve norms (orthogonal transform)."""
        x = torch.randn(8, 128)
        y = fast_hadamard_transform(x)
        # Norms should be preserved
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

    def test_different_sizes(self):
        for d in [16, 32, 64, 128, 256]:
            x = torch.randn(4, d)
            y = fast_hadamard_transform(x)
            x_recon = inverse_hadamard_transform(y)
            assert torch.allclose(x, x_recon, atol=1e-5), f"Failed for d={d}"


class TestLloydMaxCodebook:
    def test_basic_convergence(self):
        data = torch.randn(10000)
        levels, boundaries = lloyd_max_codebook(data, num_levels=8, max_iter=50)
        assert levels.shape == (8,)
        assert boundaries.shape == (7,)
        # Levels should be sorted
        assert (levels[1:] >= levels[:-1]).all()

    def test_uniform_data(self):
        data = torch.linspace(-1, 1, 10000)
        levels, _ = lloyd_max_codebook(data, num_levels=4, max_iter=50)
        # For uniform data, levels should be roughly at quartile centers
        assert levels[0] < -0.3
        assert levels[-1] > 0.3

    def test_constant_data(self):
        data = torch.ones(100) * 5.0
        levels, boundaries = lloyd_max_codebook(data, num_levels=4)
        # All levels should be 5.0
        assert torch.allclose(levels, torch.full((4,), 5.0))


class TestTurboQuantizer:
    @pytest.fixture
    def quantizer_3b(self):
        cfg = TurboQuantConfig(key_bits=3, value_bits=3, group_size=64)
        tq = TurboQuantizer(cfg)
        keys = torch.randn(4, 64, 64) * 0.5
        values = torch.randn(4, 64, 64) * 0.3
        tq.calibrate(keys, values)
        return tq

    def test_calibrate_creates_codebooks(self, quantizer_3b):
        tq = quantizer_3b
        assert tq._key_codebook is not None
        assert tq._key_codebook.shape == (8,)  # 2^3
        assert tq._value_codebook.shape == (8,)
        assert tq._calibrated

    def test_quantize_dequantize_keys(self, quantizer_3b):
        tq = quantizer_3b
        keys = torch.randn(4, 32, 64) * 0.5
        qdata = tq.quantize_keys(keys)
        recon = tq.dequantize_keys(qdata)
        assert recon.shape == keys.shape
        # Cosine similarity should be high
        cos = torch.nn.functional.cosine_similarity(
            keys.flatten().float(), recon.flatten().float(), dim=0
        )
        assert cos > 0.95

    def test_quantize_dequantize_values(self, quantizer_3b):
        tq = quantizer_3b
        values = torch.randn(4, 32, 64) * 0.3
        qdata = tq.quantize_values(values)
        recon = tq.dequantize_values(qdata)
        assert recon.shape == values.shape
        cos = torch.nn.functional.cosine_similarity(
            values.flatten().float(), recon.flatten().float(), dim=0
        )
        assert cos > 0.95

    def test_per_group_norm_stored(self, quantizer_3b):
        tq = quantizer_3b
        keys = torch.randn(4, 16, 64) * 0.5
        qdata = tq.quantize_keys(keys)
        assert 'group_mean' in qdata
        assert 'group_std' in qdata
        assert qdata['group_mean'].dtype == torch.float16
        assert qdata['group_std'].dtype == torch.float16

    def test_2bit_quantizer(self):
        cfg = TurboQuantConfig(key_bits=2, value_bits=2, group_size=64)
        tq = TurboQuantizer(cfg)
        keys = torch.randn(4, 64, 64)
        values = torch.randn(4, 64, 64)
        tq.calibrate(keys, values)
        assert tq._key_codebook.shape == (4,)  # 2^2
        qdata = tq.quantize_keys(keys)
        recon = tq.dequantize_keys(qdata)
        cos = torch.nn.functional.cosine_similarity(
            keys.flatten().float(), recon.flatten().float(), dim=0
        )
        assert cos > 0.90  # 2-bit is less precise

    def test_no_rotation(self):
        cfg = TurboQuantConfig(key_bits=3, value_bits=3, group_size=64, rotation="none")
        tq = TurboQuantizer(cfg)
        keys = torch.randn(4, 32, 64)
        values = torch.randn(4, 32, 64)
        tq.calibrate(keys, values)
        qdata = tq.quantize_keys(keys)
        recon = tq.dequantize_keys(qdata)
        assert recon.shape == keys.shape

    def test_non_power_of_2_dim(self):
        """Head dim not power of 2 should still work (padded internally)."""
        cfg = TurboQuantConfig(key_bits=3, value_bits=3, group_size=64)
        tq = TurboQuantizer(cfg)
        keys = torch.randn(4, 32, 48)  # 48 is not power of 2
        values = torch.randn(4, 32, 48)
        tq.calibrate(keys, values)
        qdata = tq.quantize_keys(keys)
        recon = tq.dequantize_keys(qdata)
        assert recon.shape == keys.shape


class TestTurboWarmTier:
    def test_basic_roundtrip(self):
        tw = TurboWarmTier(
            max_seq_len=256, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        keys = torch.randn(4, 32, 64) * 0.5
        values = torch.randn(4, 32, 64) * 0.3
        tw.quantize_and_append_kv(keys, values)
        assert tw.length == 32

        recon_k, recon_v = tw.dequantize_slice(0, 32)
        assert recon_k.shape == (4, 32, 64)
        assert recon_v.shape == (4, 32, 64)

        k_cos = torch.nn.functional.cosine_similarity(
            keys.flatten().float(), recon_k.flatten().float(), dim=0
        )
        assert k_cos > 0.95

    def test_multiple_appends(self):
        tw = TurboWarmTier(
            max_seq_len=256, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        for _ in range(4):
            keys = torch.randn(4, 16, 64)
            values = torch.randn(4, 16, 64)
            tw.quantize_and_append_kv(keys, values)
        assert tw.length == 64

        recon_k, recon_v = tw.dequantize_slice(0, 64)
        assert recon_k.shape == (4, 64, 64)

    def test_eviction_on_overflow(self):
        tw = TurboWarmTier(
            max_seq_len=64, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        # Fill to capacity
        keys = torch.randn(4, 64, 64)
        values = torch.randn(4, 64, 64)
        tw.quantize_and_append_kv(keys, values)
        assert tw.length == 64

        # Append more — should evict oldest
        new_keys = torch.randn(4, 16, 64)
        new_values = torch.randn(4, 16, 64)
        tw.quantize_and_append_kv(new_keys, new_values)
        assert tw.length == 64  # Still at capacity (evicted 16, added 16)

    def test_per_group_norm_stored(self):
        tw = TurboWarmTier(
            max_seq_len=128, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        keys = torch.randn(4, 16, 64) * 2.0  # Large variance
        values = torch.randn(4, 16, 64) * 0.1  # Small variance
        tw.quantize_and_append_kv(keys, values)

        # mean/std should be non-zero and distinct
        k_mean = tw._k_mean[:, :16, :]
        k_std = tw._k_std[:, :16, :]
        assert k_mean.abs().mean() > 0.01
        assert k_std.abs().mean() > 0.1

    def test_reset(self):
        tw = TurboWarmTier(
            max_seq_len=128, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        tw.quantize_and_append_kv(torch.randn(4, 32, 64), torch.randn(4, 32, 64))
        assert tw.length == 32
        tw.reset()
        assert tw.length == 0

    def test_bytes_used(self):
        tw = TurboWarmTier(
            max_seq_len=128, num_heads=4, head_dim=64,
            key_bits=3, value_bits=3, group_size=64, device='cpu'
        )
        assert tw.bytes_used == 0
        tw.quantize_and_append_kv(torch.randn(4, 32, 64), torch.randn(4, 32, 64))
        assert tw.bytes_used > 0
