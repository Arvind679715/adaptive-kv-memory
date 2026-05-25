import torch
from akv.turbo_quant import TurboQuantizer, TurboQuantConfig, fast_hadamard_transform

# Test Hadamard transform
x = torch.randn(4, 32, 128)
y = fast_hadamard_transform(x)
x_recon = fast_hadamard_transform(y)  # Self-inverse
print(f'Hadamard round-trip error: {(x - x_recon).abs().max().item():.2e}')

# Test TurboQuantizer
cfg = TurboQuantConfig(key_bits=3, value_bits=2, group_size=128, rotation='hadamard')
tq = TurboQuantizer(cfg)

# Generate sample KV data (simulating real activations with outliers)
keys = torch.randn(8, 256, 128) * 0.5
keys[:, :10, :5] = 5.0  # Simulate outlier channels
values = torch.randn(8, 256, 128) * 0.3

# Calibrate
tq.calibrate(keys, values)
print(f'Key codebook ({cfg.key_bits}b): {tq._key_codebook}')
print(f'Value codebook ({cfg.value_bits}b): {tq._value_codebook}')

# Quantize and measure
metrics = tq.quantize_and_measure(keys, values)
print(f'Key cosine: {metrics["key_cosine"]:.6f}')
print(f'Value cosine: {metrics["value_cosine"]:.6f}')
print(f'Compression: {metrics["compression_ratio"]:.1f}x')
print(f'Key MSE: {metrics["key_mse"]:.6f}')
print(f'Value MSE: {metrics["value_mse"]:.6f}')

# Compare with simple min-max 
from akv.quantizer import KVQuantizer, QuantConfig
baseline = KVQuantizer(QuantConfig(bits=4, group_size=128))
q_baseline = baseline.quantize(keys)
recon_baseline = baseline.dequantize(q_baseline)
baseline_cos = torch.nn.functional.cosine_similarity(keys.flatten(), recon_baseline.flatten(), dim=0).item()
print(f'\nBaseline 4-bit min-max cosine: {baseline_cos:.6f}')
print(f'TurboQuant 3-bit key cosine:   {metrics["key_cosine"]:.6f}')
print(f'TurboQuant 2-bit val cosine:   {metrics["value_cosine"]:.6f}')
print(f'\nBaseline compression: {q_baseline.compression_ratio:.1f}x')
print(f'TurboQuant compression: {metrics["compression_ratio"]:.1f}x')
