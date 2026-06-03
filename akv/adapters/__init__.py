"""Per-architecture adapters for AKV.

Different model families have quirks that affect how a KV cache must behave:

* **Llama / TinyLlama / Llama-2 / Llama-3**: standard MHA/GQA + RoPE. Reference.
* **Mistral / Mixtral**: sliding-window attention — recent N tokens are special;
  tokens evicted past the window are unreachable, which interacts with AKV's
  promotion logic.
* **Qwen2 / Qwen2.5**: GQA, very small KV-head count (2 for 0.5B). Importance
  scoring per KV-group still works but bit allocation must be group-aware.
* **Gemma / Gemma-2**: logit softcapping + (in gemma-2) alternating local/global
  attention layers. Local layers shouldn't be tiered.
* **Phi-3 / Phi-3.5**: standard MHA with rope_scaling. Works out of the box,
  but recent versions exposed quirks in DynamicCache reordering.
* **DeepSeek-V2 MLA**: multi-head latent attention compresses KV into a small
  latent vector — our quantization is redundant, so we no-op.

The adapter registry maps a HuggingFace `model_type` -> an `AdapterSpec`
describing how AKV should configure itself. Users never need to call this
directly: `AKVCache.for_model(model)` consults the registry automatically.
"""
from __future__ import annotations

from akv.adapters.registry import (
    AdapterSpec,
    get_adapter,
    list_adapters,
    register_adapter,
    resolve_for_model,
)

__all__ = [
    "AdapterSpec",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "resolve_for_model",
]
