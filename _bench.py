import torch
import torch.nn.functional as F
from akv.turbo_quant import TurboQuantizer, TurboQuantConfig
from akv.quantizer import KVQuantizer, QuantConfig

torch.manual_seed(42)

H, N, D = 32, 512, 128
keys = torch.randn(H, N, D, dtype=torch.float32) * 0.3
values = torch.randn(H, N, D, dtype=torch.float32) * 0.2

outlier_channels = [3, 17, 45, 89, 112]
keys[:, :, outlier_channels] *= 15.0
values[:, :, outlier_channels] *= 8.0

print('=== Quality Comparison: NormQuant vs Min-Max ===')
print(f'Data: {H} heads, {N} tokens, {D} dim')
print(f'Outlier channels: {outlier_channels} (15x key, 8x value magnitude)')
print()

mm4 = KVQuantizer(QuantConfig(bits=4, group_size=128))
q_k4 = mm4.quantize(keys.half())
recon_k4 = mm4.dequantize(q_k4).float()
q_v4 = mm4.quantize(values.half())
recon_v4 = mm4.dequantize(q_v4).float()

k_cos_mm4 = F.cosine_similarity(keys.flatten(), recon_k4.flatten(), dim=0).item()
v_cos_mm4 = F.cosine_similarity(values.flatten(), recon_v4.flatten(), dim=0).item()
k_mse_mm4 = ((keys - recon_k4)**2).mean().item()
v_mse_mm4 = ((values - recon_v4)**2).mean().item()

mm2 = KVQuantizer(QuantConfig(bits=2, group_size=128))
q_k2 = mm2.quantize(keys.half())
recon_k2 = mm2.dequantize(q_k2).float()
q_v2 = mm2.quantize(values.half())
recon_v2 = mm2.dequantize(q_v2).float()

k_cos_mm2 = F.cosine_similarity(keys.flatten(), recon_k2.flatten(), dim=0).item()
v_cos_mm2 = F.cosine_similarity(values.flatten(), recon_v2.flatten(), dim=0).item()
k_mse_mm2 = ((keys - recon_k2)**2).mean().item()
v_mse_mm2 = ((values - recon_v2)**2).mean().item()

tq = TurboQuantizer(TurboQuantConfig(key_bits=3, value_bits=2, group_size=128, rotation='hadamard'))
tq.calibrate(keys, values)
metrics = tq.quantize_and_measure(keys, values)

tq43 = TurboQuantizer(TurboQuantConfig(key_bits=4, value_bits=3, group_size=128, rotation='hadamard'))
tq43.calibrate(keys, values)
metrics43 = tq43.quantize_and_measure(keys, values)

print(f"{'Method':<25} {'K-cos':>8} {'V-cos':>8} {'K-MSE':>10} {'V-MSE':>10} {'Comp':>6}")
print('-' * 75)
print(f"{'Min-Max 4-bit (current)':<25} {k_cos_mm4:>8.6f} {v_cos_mm4:>8.6f} {k_mse_mm4:>10.6f} {v_mse_mm4:>10.6f} {'3.6x':>6}")
print(f"{'Min-Max 2-bit':<25} {k_cos_mm2:>8.6f} {v_cos_mm2:>8.6f} {k_mse_mm2:>10.6f} {v_mse_mm2:>10.6f} {'7.3x':>6}")
print(f"{'NormQuant 3b-K/2b-V':<25} {metrics['key_cosine']:>8.6f} {metrics['value_cosine']:>8.6f} {metrics['key_mse']:>10.6f} {metrics['value_mse']:>10.6f} {metrics['compression_ratio']:>5.1f}x")
print(f"{'NormQuant 4b-K/3b-V':<25} {metrics43['key_cosine']:>8.6f} {metrics43['value_cosine']:>8.6f} {metrics43['key_mse']:>10.6f} {metrics43['value_mse']:>10.6f} {metrics43['compression_ratio']:>5.1f}x")
print()

print('=== Key Insight ===')
print(f"Min-Max 4-bit key cosine:    {k_cos_mm4:.6f} (3.6x compression)")
print(f"NormQuant 3b key cosine:    {metrics['key_cosine']:.6f} ({metrics['compression_ratio']:.1f}x compression)")
print(f"NormQuant 4b key cosine:    {metrics43['key_cosine']:.6f} ({metrics43['compression_ratio']:.1f}x compression)")
print()
if metrics['key_cosine'] > k_cos_mm2:
    print('NormQuant 3b keys BEATS min-max 2b keys with better cosine!')
if metrics43['key_cosine'] > k_cos_mm4:
    print('NormQuant 4b keys BEATS min-max 4b keys -- same bits, better quality!')
