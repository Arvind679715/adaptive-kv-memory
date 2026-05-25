# Architecture Diagrams

## System Architecture (Mermaid)

```mermaid
graph TB
    subgraph "Adaptive KV Memory Engine"
        direction TB

        subgraph "Input Layer"
            QKV[Q, K, V from Attention]
            AW[Attention Weights]
        end

        subgraph "Core Engine"
            IS[Importance Scorer<br/>Hybrid: Attention + Recency]
            AE[Adaptive Evictor<br/>Budget-Aware Policy]
            QZ[KV Quantizer<br/>Group-wise Asymmetric]
            TK[Triton Fused Kernels<br/>Mixed-Precision Attention]
        end

        subgraph "Three-Tier Memory Hierarchy"
            direction LR
            HOT[🔥 Hot Tier<br/>GPU HBM • FP16<br/>1024 tokens]
            WARM[⚡ Warm Tier<br/>GPU HBM • INT4<br/>2048 tokens]
            COLD[❄️ Cold Tier<br/>CPU RAM • INT2<br/>Unlimited]
        end

        subgraph "Integration Layer"
            HF[HuggingFace<br/>DynamicCache API]
            VLLM[vLLM<br/>CacheEngine]
            TRT[TensorRT-LLM<br/>Plugin API]
        end
    end

    QKV --> IS
    AW --> IS
    IS --> AE
    AE --> HOT
    HOT -- "Demote (quantize)" --> WARM
    WARM -- "Demote (compress)" --> COLD
    COLD -- "Promote (decompress)" --> WARM
    WARM -- "Promote (dequantize)" --> HOT
    QZ --> WARM
    QZ --> COLD
    TK --> HOT
    TK --> WARM
    HOT --> HF
    HOT --> VLLM
    HOT --> TRT
```

## Data Flow During Inference

```mermaid
sequenceDiagram
    participant M as Model Layer
    participant C as Cache Manager
    participant S as Importance Scorer
    participant E as Evictor
    participant H as Hot Tier (GPU/FP16)
    participant W as Warm Tier (GPU/INT4)
    participant K as Cold Tier (CPU/INT2)

    M->>C: update(K, V, layer_idx, attn_weights)
    C->>S: score(attn_weights, layer_idx)
    S->>S: decay old scores + accumulate new

    C->>H: append new KV pairs
    C->>C: check budget: hot_len > hot_budget?

    alt Hot tier over budget
        C->>E: compute_eviction(scores, hot_budget)
        E->>E: select lowest-importance tokens
        E-->>C: evict_indices, keep_indices
        C->>W: quantize & move evicted tokens
        Note over W: 4-bit group quantization
    end

    alt Warm tier over budget
        C->>W: identify coldest tokens
        C->>K: quantize to 2-bit & move to CPU
        Note over K: Async transfer via CUDA stream
    end

    M->>C: get KV for attention
    C->>H: return hot KV (fp16, fast path)
    C->>W: fused dequant in attention kernel
    Note over H,W: Single fused softmax across both tiers
    C-->>M: attention output
```

## Memory Layout

```mermaid
graph LR
    subgraph "GPU HBM (24GB)"
        subgraph "Model Weights (~14GB)"
            MW[Parameters]
        end
        subgraph "Hot Tier (~2GB)"
            HK[Hot Keys<br/>B×H×1024×D fp16]
            HV[Hot Values<br/>B×H×1024×D fp16]
            HP[Hot Positions<br/>1024 × int32]
        end
        subgraph "Warm Tier (~1GB)"
            WK[Warm Keys<br/>B×H×2048×D packed int4]
            WV[Warm Values<br/>B×H×2048×D packed int4]
            WS[Scales/Zeros<br/>per-group fp16]
        end
        ACT[Activations<br/>~2GB]
    end

    subgraph "CPU RAM"
        subgraph "Cold Tier"
            CK[Cold Keys<br/>packed int2]
            CV[Cold Values<br/>packed int2]
            CS[Scales/Zeros]
        end
    end

    style HK fill:#ff6b6b
    style HV fill:#ff6b6b
    style WK fill:#feca57
    style WV fill:#feca57
    style CK fill:#48dbfb
    style CV fill:#48dbfb
```

## Benchmark Comparison Architecture

```mermaid
graph TB
    subgraph "Evaluation Framework"
        direction TB
        BM[Benchmark Manager]

        subgraph "Methods Under Test"
            FC[Full Cache<br/>No compression]
            H2O[H2O<br/>Heavy-Hitter Oracle]
            KIVI[KIVI<br/>Uniform 2-bit]
            SNP[SnapKV<br/>Observation window]
            SCH[ScissorHands<br/>Persistence filter]
            AKV[AKV (Ours)<br/>Hierarchical adaptive]
        end

        subgraph "Benchmark Suite"
            TP[Throughput<br/>tok/s at scale]
            LT[Latency<br/>TTFT, ITL p50/p99]
            PP[Perplexity<br/>WikiText-2, PG-19]
            NI[Needle-in-Haystack<br/>Delayed recall]
            MN[Multi-Needle<br/>Distributed facts]
            MS[Memory Scaling<br/>VRAM vs seq_len]
        end

        subgraph "Metrics"
            QM[Quality<br/>PPL ratio ≤ 1.02]
            SM[Speed<br/>tok/s, TTFT]
            MM[Memory<br/>VRAM savings %]
            RM[Recall<br/>Accuracy @ depth]
        end
    end

    BM --> FC & H2O & KIVI & SNP & SCH & AKV
    FC & H2O & KIVI & SNP & SCH & AKV --> TP & LT & PP & NI & MN & MS
    TP & LT & PP & NI & MN & MS --> QM & SM & MM & RM
```

## Triton Kernel Fusion Strategy

```mermaid
graph LR
    subgraph "Standard Approach (Wasteful)"
        direction TB
        D1[Dequantize K<br/>N×D int4→fp16] --> A1[Q × K^T<br/>M×N matmul]
        A1 --> S1[Softmax]
        D2[Dequantize V<br/>N×D int4→fp16] --> A2[Attn × V<br/>M×N × N×D]
        S1 --> A2
        style D1 fill:#ff6b6b
        style D2 fill:#ff6b6b
    end

    subgraph "Our Fused Approach"
        direction TB
        F1[Fused Dequant+Dot<br/>Q × dequant(K)^T<br/>On-the-fly per tile] --> F2[Online Softmax<br/>Streaming, no materialization]
        F2 --> F3[Fused Attn×dequant(V)<br/>Tile-by-tile accumulation]
        style F1 fill:#00d2d3
        style F2 fill:#00d2d3
        style F3 fill:#00d2d3
    end
```

## Integration Architecture

```mermaid
graph TB
    subgraph "User-Facing APIs"
        API1["gen = AdaptiveGenerator(model, tok)<br/>output = gen.generate(prompt)"]
        API2["pipe = adaptive_pipeline('text-generation')<br/>result = pipe(prompt)"]
        API3["llm = AdaptiveKVLLM(model)<br/>outputs = llm.generate(prompts)"]
    end

    subgraph "Integration Layer"
        HFI[HFAdaptiveCache<br/>DynamicCache compatible]
        VLLMI[AdaptiveCacheEngine<br/>vLLM CacheEngine compatible]
    end

    subgraph "Core AKV Engine"
        AKV[AdaptiveKVCache]
        IS[ImportanceScorer]
        AE[AdaptiveEvictor]
        QZ[KVQuantizer]
        TO[Triton Ops]
    end

    API1 --> HFI
    API2 --> HFI
    API3 --> VLLMI
    HFI --> AKV
    VLLMI --> AKV
    AKV --> IS & AE & QZ & TO
```
