from akv.production_cache import ProductionCache, ProductionCacheConfig
import torch
cfg = ProductionCacheConfig(num_layers=1, num_heads=4, head_dim=64, hot_budget=32, warm_budget=128, page_size=8, max_hot_pages=32, device='cpu')
cache = ProductionCache(cfg)
k = torch.randn(1,4,64,64)
v = torch.randn(1,4,64,64)
attn = torch.rand(1,4,64,64)
attn = attn / attn.sum(-1, keepdim=True)
cache.update(k, v, 0, attention_weights=attn)
print(f'hot={cache._layers[0].hot_len}, warm={cache._layers[0].warm_len}')
[cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0) for _ in range(40)]
print(f'hot={cache._layers[0].hot_len}, warm={cache._layers[0].warm_len}')
q = torch.randn(1,4,1,64)
out = cache.fused_attention(q, 0)
print(f'Attention output: {out.shape}')
usage = cache.memory_usage()
print(f'Migrations: {usage["migrations"]}, demoted: {usage["total_demoted"]}')
