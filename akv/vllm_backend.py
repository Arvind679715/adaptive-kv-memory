"""Stable shim that matches the upstream vLLM PR contract.

This module exists so that the integration test code in
`docs/vllm_pr.md` works today (against the in-process patcher) and will
keep working unchanged once vLLM merges the `kv_cache_backend="akv"`
option. See docs/vllm_pr.md for the PR design.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from akv.vllm_integration import (
    AdaptiveCacheEngine,
    AdaptiveKVLLM,
    AdaptiveVLLMConfig,
    patch_vllm_model_runner,
)

logger = logging.getLogger(__name__)

__all__ = [
    "create_akv_llm",
    "AdaptiveKVLLM",
    "AdaptiveCacheEngine",
    "AdaptiveVLLMConfig",
]


def create_akv_llm(
    model: str,
    *,
    hot_budget_per_seq: int = 1024,
    warm_budget_per_seq: int = 4096,
    warm_bits: int = 3,
    enable_cold_tier: bool = True,
    **vllm_kwargs: Any,
) -> AdaptiveKVLLM:
    """Convenience builder mirroring the proposed upstream signature.

    Once vLLM merges `kv_cache_backend="akv"`, users can replace::

        from akv.vllm_backend import create_akv_llm
        llm = create_akv_llm("meta-llama/Llama-3.1-8B-Instruct")

    with::

        from vllm import LLM
        llm = LLM("meta-llama/Llama-3.1-8B-Instruct", kv_cache_backend="akv")

    and everything else stays the same.
    """
    cfg = AdaptiveVLLMConfig(
        hot_budget_per_seq=hot_budget_per_seq,
        warm_budget_per_seq=warm_budget_per_seq,
        warm_bits=warm_bits,
        enable_cold_tier=enable_cold_tier,
    )
    return AdaptiveKVLLM(model=model, adaptive_config=cfg, **vllm_kwargs)
