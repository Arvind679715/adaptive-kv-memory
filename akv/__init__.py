"""Adaptive KV Memory — retrieval-preserving hierarchical KV cache compression for LLMs.

Quick start:
    from akv import ProductionCache, ProductionCacheConfig

    config = ProductionCacheConfig(
        num_layers=32, num_heads=32, head_dim=128,
        hot_budget=1024, warm_budget=4096, warm_bits=3,
    )
    cache = ProductionCache(config)
"""
__version__ = "1.0.0"

# --- Core API (what most users need) ---
from akv.production_cache import ProductionCache, ProductionCacheConfig
from akv.hf_generate import AdaptiveGenerator, GeneratorConfig, adaptive_pipeline
from akv.turbo_quant import TurboQuantizer, TurboQuantConfig

# --- Research/evaluation API ---
from akv.quantizer import KVQuantizer, QuantConfig
from akv.importance import ImportanceScorer, ImportanceConfig
from akv.evictor import AdaptiveEvictor, EvictionConfig
from akv.cache import AdaptiveKVCache, CacheConfig
from akv.integration import HFAdaptiveCache, HFProductionCache
from akv.baselines import (
    FullCache, H2OCache, H2OConfig,
    KIVICache, KIVIConfig, SnapKVCache, SnapKVConfig,
    ScissorHandsCache, ScissorHandsConfig, create_baseline,
)
from akv.triton_ops import (
    fused_quantized_attention, fused_mixed_precision_attention,
    fused_importance_update, HAS_TRITON,
)
from akv.triton_kernels import (
    fused_decode_attention, fused_quantize_evict,
)
from akv.evaluation import (
    PerplexityEvaluator, EvalConfig, MethodConfig,
    measure_memory_scaling, run_ablation, get_standard_methods,
)
from akv.async_migration import AsyncMigrator, AsyncMigratorConfig
from akv.prefetch import PrefetchScheduler, PrefetchConfig
from akv.packed_layout import PackedKVArena, PackedKVConfig, PagedKVCache
from akv.fused_attention import (
    fused_int4_decode_attention, mixed_precision_decode_attention,
)

__all__ = [
    # Core
    "ProductionCache", "ProductionCacheConfig",
    "AdaptiveGenerator", "GeneratorConfig", "adaptive_pipeline",
    "TurboQuantizer", "TurboQuantConfig",
    # Research
    "AdaptiveKVCache", "CacheConfig",
    "KVQuantizer", "QuantConfig",
    "ImportanceScorer", "ImportanceConfig",
    "AdaptiveEvictor", "EvictionConfig",
    "HFAdaptiveCache",
    # Baselines
    "FullCache", "H2OCache", "KIVICache", "SnapKVCache",
    "ScissorHandsCache", "create_baseline",
    # Kernels
    "HAS_TRITON", "fused_quantized_attention",
    "fused_decode_attention", "fused_quantize_evict",
    # Evaluation
    "PerplexityEvaluator", "EvalConfig",
]
