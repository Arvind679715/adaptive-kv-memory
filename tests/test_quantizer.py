"""Tests for KV quantizer."""
import pytest
import torch
from akv.quantizer import KVQuantizer, QuantConfig, QuantizedTensor


class TestQuantConfig:
    def test_default_config(self):
        cfg = QuantConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.symmetric is False

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be"):
            QuantConfig(bits=3)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size must be"):
            QuantConfig(group_size=0)


class TestQuantizerRoundtrip:
    """Test that quantize -> dequantize preserves tensor shape and approximate values."""

    @pytest.fixture
    def quantizer(self):
        return KVQuantizer(QuantConfig(bits=4, group_size=32))

    @pytest.fixture
    def sample_kv(self):
        # Typical KV shape: (batch=1, heads=8, seq_len=64, head_dim=64)
        torch.manual_seed(42)
        return torch.randn(1, 8, 64, 64, dtype=torch.float16)

    def test_4bit_roundtrip_shape(self, quantizer, sample_kv):
        qtensor = quantizer.quantize(sample_kv)
        reconstructed = quantizer.dequantize(qtensor)
        assert reconstructed.shape == sample_kv.shape

    def test_4bit_roundtrip_error(self, quantizer, sample_kv):
        qtensor = quantizer.quantize(sample_kv)
        reconstructed = quantizer.dequantize(qtensor)
        mse = (sample_kv.float() - reconstructed.float()).pow(2).mean().item()
        # 4-bit quantization should have reasonable error
        assert mse < 0.1, f"MSE too high: {mse}"

    def test_8bit_roundtrip(self, sample_kv):
        q = KVQuantizer(QuantConfig(bits=8, group_size=32))
        qtensor = q.quantize(sample_kv)
        reconstructed = q.dequantize(qtensor)
        assert reconstructed.shape == sample_kv.shape
        mse = (sample_kv.float() - reconstructed.float()).pow(2).mean().item()
        assert mse < 0.01, f"8-bit MSE too high: {mse}"

    def test_2bit_roundtrip(self, sample_kv):
        q = KVQuantizer(QuantConfig(bits=2, group_size=32))
        qtensor = q.quantize(sample_kv)
        reconstructed = q.dequantize(qtensor)
        assert reconstructed.shape == sample_kv.shape
        # 2-bit will have higher error, but shape must be correct
        mse = (sample_kv.float() - reconstructed.float()).pow(2).mean().item()
        assert mse < 1.0, f"2-bit MSE unexpectedly high: {mse}"

    def test_compression_ratio(self, quantizer, sample_kv):
        qtensor = quantizer.quantize(sample_kv)
        assert qtensor.compression_ratio > 1.0
        # 4-bit should give ~4x compression (minus overhead)
        assert qtensor.compression_ratio > 2.0

    def test_different_group_sizes(self, sample_kv):
        for gs in [16, 32, 64, 128]:
            q = KVQuantizer(QuantConfig(bits=4, group_size=gs))
            qtensor = q.quantize(sample_kv)
            reconstructed = q.dequantize(qtensor)
            assert reconstructed.shape == sample_kv.shape

    def test_symmetric_quantization(self, sample_kv):
        q = KVQuantizer(QuantConfig(bits=4, group_size=32, symmetric=True))
        qtensor = q.quantize(sample_kv)
        reconstructed = q.dequantize(qtensor)
        assert reconstructed.shape == sample_kv.shape
        mse = (sample_kv.float() - reconstructed.float()).pow(2).mean().item()
        # Symmetric quant has higher error than asymmetric (wastes range)
        assert mse < 0.5


class TestPackUnpack:
    def test_4bit_pack_unpack(self):
        q = KVQuantizer()
        data = torch.tensor([3, 7, 1, 15, 0, 8], dtype=torch.uint8)
        packed = q._pack(data, 4)
        unpacked = q._unpack(packed, 4, 6)
        assert torch.allclose(data.float(), unpacked)

    def test_2bit_pack_unpack(self):
        q = KVQuantizer()
        data = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2], dtype=torch.uint8)
        packed = q._pack(data, 2)
        unpacked = q._unpack(packed, 2, 8)
        assert torch.allclose(data.float(), unpacked)

    def test_8bit_passthrough(self):
        q = KVQuantizer()
        data = torch.tensor([0, 128, 255, 42], dtype=torch.uint8)
        packed = q._pack(data, 8)
        unpacked = q._unpack(packed, 8, 4)
        assert torch.allclose(data.float(), unpacked)


class TestEstimateError:
    def test_higher_bits_lower_error(self):
        torch.manual_seed(42)
        tensor = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        q = KVQuantizer(QuantConfig(group_size=32))
        err_2 = q.estimate_error(tensor, bits=2)
        err_4 = q.estimate_error(tensor, bits=4)
        err_8 = q.estimate_error(tensor, bits=8)
        assert err_8 < err_4 < err_2


class TestMemorySavings:
    def test_estimate(self):
        q = KVQuantizer(QuantConfig(bits=4, group_size=128))
        savings = q.estimate_memory_savings(
            num_layers=32, num_heads=32, seq_len=4096, head_dim=128, bits=4
        )
        assert savings["original_mb"] > 0
        assert savings["quantized_mb"] > 0
        assert savings["compression_ratio"] > 2.0
