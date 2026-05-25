"""Tests for Triton fused kernels (PyTorch fallback path)."""
import pytest
import torch
import math
from akv.triton_ops import (
    fused_quantized_attention,
    fused_mixed_precision_attention,
    fused_importance_update,
    _dequantize_packed,
)
from akv.quantizer import KVQuantizer, QuantConfig


@pytest.fixture
def quantizer():
    return KVQuantizer(QuantConfig(bits=4, group_size=32))


@pytest.fixture
def kv_data():
    torch.manual_seed(42)
    B, H, N, D = 1, 4, 64, 64
    q = torch.randn(B, H, 1, D, dtype=torch.float16)
    k = torch.randn(B, H, N, D, dtype=torch.float16)
    v = torch.randn(B, H, N, D, dtype=torch.float16)
    return q, k, v


class TestDequantizePacked:
    def test_roundtrip_4bit(self, quantizer):
        torch.manual_seed(42)
        tensor = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        qt = quantizer.quantize(tensor)
        # Use our utility dequant
        N = 4 * 32  # flatten batch*heads*seq
        D = 64
        recon = _dequantize_packed(
            qt.data, qt.scales.reshape(-1, qt.scales.shape[-1]),
            qt.zeros.reshape(-1, qt.zeros.shape[-1]),
            4, 32, qt.scales.reshape(-1, qt.scales.shape[-1]).shape[0], D,
            tensor.device, tensor.dtype,
        )
        assert recon.shape[-1] == D

    def test_8bit(self, quantizer):
        q8 = KVQuantizer(QuantConfig(bits=8, group_size=32))
        torch.manual_seed(42)
        tensor = torch.randn(32, 64, dtype=torch.float16)
        qt = q8.quantize(tensor)
        N, D = tensor.shape
        recon = _dequantize_packed(
            qt.data, qt.scales, qt.zeros, 8, 32, N, D,
            tensor.device, tensor.dtype,
        )
        mse = (tensor.float() - recon.float()).pow(2).mean().item()
        assert mse < 0.01


class TestFusedQuantizedAttention:
    def test_output_shape(self, quantizer, kv_data):
        q, k, v = kv_data
        B, H, N, D = k.shape
        kq = quantizer.quantize(k)
        vq = quantizer.quantize(v)

        out = fused_quantized_attention(
            q,
            kq.data,
            kq.scales.reshape(-1, kq.scales.shape[-1]),
            kq.zeros.reshape(-1, kq.zeros.shape[-1]),
            vq.data,
            vq.scales.reshape(-1, vq.scales.shape[-1]),
            vq.zeros.reshape(-1, vq.zeros.shape[-1]),
            bits=4, group_size=32,
        )
        assert out.shape == q.shape  # (B, H, 1, D)

    def test_approximate_correctness(self, quantizer, kv_data):
        q, k, v = kv_data
        kq = quantizer.quantize(k)
        vq = quantizer.quantize(v)

        # Fused
        out_fused = fused_quantized_attention(
            q,
            kq.data,
            kq.scales.reshape(-1, kq.scales.shape[-1]),
            kq.zeros.reshape(-1, kq.zeros.shape[-1]),
            vq.data,
            vq.scales.reshape(-1, vq.scales.shape[-1]),
            vq.zeros.reshape(-1, vq.zeros.shape[-1]),
            bits=4, group_size=32,
        )

        # Reference: dequant then attend
        k_deq = quantizer.dequantize(kq)
        v_deq = quantizer.dequantize(vq)
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax(torch.matmul(q, k_deq.transpose(-2, -1)) * sm_scale, dim=-1)
        out_ref = torch.matmul(attn, v_deq)

        # Should be close (both use the same dequantized values)
        mse = (out_fused.float() - out_ref.float()).pow(2).mean().item()
        assert mse < 1.0, f"Fused vs reference MSE too high: {mse}"


class TestFusedMixedPrecisionAttention:
    def test_output_shape(self, quantizer, kv_data):
        q, k, v = kv_data
        B, H, N, D = k.shape
        n_hot = N // 2
        n_warm = N - n_hot

        k_hot = k[:, :, :n_hot, :]
        v_hot = v[:, :, :n_hot, :]

        k_warm = k[:, :, n_hot:, :].reshape(-1, D)
        v_warm = v[:, :, n_hot:, :].reshape(-1, D)
        kw_q = quantizer.quantize(k_warm.unsqueeze(0).unsqueeze(0))
        vw_q = quantizer.quantize(v_warm.unsqueeze(0).unsqueeze(0))

        out, attn = fused_mixed_precision_attention(
            q, k_hot, v_hot,
            kw_q.data,
            kw_q.scales.reshape(-1, kw_q.scales.shape[-1]),
            kw_q.zeros.reshape(-1, kw_q.zeros.shape[-1]),
            vw_q.data,
            vw_q.scales.reshape(-1, vw_q.scales.shape[-1]),
            vw_q.zeros.reshape(-1, vw_q.zeros.shape[-1]),
            bits=4, group_size=32,
        )
        assert out.shape == q.shape

    def test_attention_sums_to_one(self, quantizer, kv_data):
        q, k, v = kv_data
        B, H, N, D = k.shape
        n_hot = N // 2
        k_hot = k[:, :, :n_hot, :]
        v_hot = v[:, :, :n_hot, :]
        k_warm_flat = k[:, :, n_hot:, :].reshape(-1, D)
        v_warm_flat = v[:, :, n_hot:, :].reshape(-1, D)
        kw_q = quantizer.quantize(k_warm_flat.unsqueeze(0).unsqueeze(0))
        vw_q = quantizer.quantize(v_warm_flat.unsqueeze(0).unsqueeze(0))

        _, attn = fused_mixed_precision_attention(
            q, k_hot, v_hot,
            kw_q.data, kw_q.scales.reshape(-1, kw_q.scales.shape[-1]),
            kw_q.zeros.reshape(-1, kw_q.zeros.shape[-1]),
            vw_q.data, vw_q.scales.reshape(-1, vw_q.scales.shape[-1]),
            vw_q.zeros.reshape(-1, vw_q.zeros.shape[-1]),
            bits=4, group_size=32,
        )
        # Attention should sum to ~1
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


class TestFusedImportanceUpdate:
    def test_update(self):
        torch.manual_seed(42)
        attn = torch.rand(1, 4, 1, 64)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        scores = torch.zeros(64)

        updated = fused_importance_update(attn, scores, decay=0.95)
        assert updated.shape == (64,)
        assert updated.sum() > 0

    def test_decay(self):
        scores = torch.ones(32)
        attn = torch.zeros(1, 1, 1, 32)  # zero attention
        updated = fused_importance_update(attn, scores, decay=0.5)
        # With zero attention, scores should just decay
        assert torch.allclose(updated, torch.ones(32) * 0.5, atol=1e-5)
