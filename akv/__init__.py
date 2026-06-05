"""AKV — Virtual Memory System for LLM KV Caches.

Drop-in replacement for any HuggingFace model's KV cache:

    from akv import AKVCache
    cache = AKVCache(preset="balanced")
    outputs = model(**inputs, past_key_values=cache, use_cache=True)

That's it. No model surgery, no custom attention, no monkey-patching.

For model-aware setup:
    cache = AKVCache.for_model(model, preset="balanced",
                               protect_first=2, protect_last=2)
"""
__version__ = "1.2.0"

# --- Drop-in API (what most users need) ---
from akv.drop_in import AKVCache, AKVLayer, recommend_preset

# --- Calibration & adapters (production setup) ---
from akv.calibration import CalibrationReport, HeadSensitivity, calibrate_model
from akv.adapters import (
    AdapterSpec, get_adapter, list_adapters, register_adapter, resolve_for_model,
)

# --- Production API (for serving systems) ---
from akv.production_cache import ProductionCache, ProductionCacheConfig
from akv.hf_generate import AdaptiveGenerator, GeneratorConfig, adaptive_pipeline
from akv.turbo_quant import NormQuantizer, NormQuantConfig, TurboQuantizer, TurboQuantConfig
from akv.diagnostics import recommend_preset as diagnose_preset, diagnose_model, DiagnosticReport

# --- Research/evaluation API ---
from akv.quantizer import KVQuantizer, QuantConfig
from akv.importance import ImportanceScorer, ImportanceConfig
from akv.evictor import AdaptiveEvictor, EvictionConfig
from akv.cache import AdaptiveKVCache, CacheConfig
from akv.integration import HFAdaptiveCache, HFProductionCache
from akv.baselines import (
    FullCache, H2OCache, H2OConfig,
    KIVICache, KIVIConfig, KIVIFusedCache, KIVIFusedConfig,
    SnapKVCache, SnapKVConfig,
    ScissorHandsCache, ScissorHandsConfig,
    StreamingLLMCache, StreamingLLMConfig,
    PyramidKVCache, PyramidKVConfig,
    create_baseline,
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
from akv.vmm import (
    VirtualMemoryCache, VMMConfig,
    ImportanceBasedMigrator, RetrievalAwarePromoter,
    AdaptiveBitAllocator, PagedVirtualMemory, FlashAttentionAdapter,
    MigrationPolicy, PromotionPolicy,
)

__all__ = [
    # Drop-in (recommended for most users)
    "AKVCache", "AKVLayer", "recommend_preset",
    # Calibration + adapters
    "calibrate_model", "CalibrationReport", "HeadSensitivity",
    "AdapterSpec", "get_adapter", "list_adapters",
    "register_adapter", "resolve_for_model",
    # Production serving
    "ProductionCache", "ProductionCacheConfig",
    "AdaptiveGenerator", "GeneratorConfig", "adaptive_pipeline",
    "TurboQuantizer", "TurboQuantConfig",
    "diagnose_model", "diagnose_preset", "DiagnosticReport",
    # Virtual Memory Manager (VMM)
    "VirtualMemoryCache", "VMMConfig",
    "ImportanceBasedMigrator", "RetrievalAwarePromoter",
    "AdaptiveBitAllocator", "PagedVirtualMemory", "FlashAttentionAdapter",
    "MigrationPolicy", "PromotionPolicy",
    # Research
    "AdaptiveKVCache", "CacheConfig",
    "KVQuantizer", "QuantConfig",
    "ImportanceScorer", "ImportanceConfig",
    "AdaptiveEvictor", "EvictionConfig",
    "HFAdaptiveCache",
    # Baselines
    "FullCache", "H2OCache", "KIVICache", "KIVIFusedCache", "SnapKVCache",
    "ScissorHandsCache", "create_baseline",
    # Kernels
    "HAS_TRITON", "fused_quantized_attention",
    "fused_decode_attention", "fused_quantize_evict",
    # Evaluation
    "PerplexityEvaluator", "EvalConfig",
]
