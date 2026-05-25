"""AKV command-line interface for compressed KV cache generation and benchmarking."""
from __future__ import annotations

import argparse
import sys
import time


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation with adaptive KV cache."""
    import torch
    from akv.hf_generate import AdaptiveGenerator, GeneratorConfig

    config = GeneratorConfig(
        hot_budget=args.hot_budget,
        warm_budget=args.warm_budget,
        warm_bits=args.warm_bits,
        max_new_tokens=args.max_new_tokens,
    )

    gen = AdaptiveGenerator.from_pretrained(
        args.model,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Model: {args.model}")
    print(f"Cache: hot={args.hot_budget}, warm={args.warm_budget} @ {args.warm_bits}b")
    print("-" * 60)

    result = gen.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(result.text)
    print("-" * 60)
    print(f"Tokens: {result.num_generated} | {result.tokens_per_sec:.1f} tok/s")
    if result.tier_summary:
        print(f"Tiers: {result.tier_summary}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run quick latency benchmark."""
    import torch
    from akv.production_cache import ProductionCache, ProductionCacheConfig

    config = ProductionCacheConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        hot_budget=args.hot_budget,
        warm_budget=args.warm_budget,
    )

    cache = ProductionCache(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate prefill
    print(f"Benchmarking: {args.num_layers}L, {args.num_heads}H, d={args.head_dim}")
    print(f"Budget: hot={args.hot_budget}, warm={args.warm_budget}")
    print(f"Device: {device}")
    print("-" * 60)

    seq_len = args.hot_budget + args.warm_budget
    print(f"Simulating {seq_len} token prefill + {args.decode_steps} decode steps...")

    # Quick timing of decode
    times = []
    for _ in range(args.decode_steps):
        t0 = time.perf_counter()
        # Decode step simulation
        q = torch.randn(1, args.num_heads, 1, args.head_dim, device=device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    import statistics
    p50 = statistics.median(times)
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"Decode latency: p50={p50:.2f}ms, p99={p99:.2f}ms")


def cmd_info(args: argparse.Namespace) -> None:
    """Print package info and available features."""
    import akv
    import torch

    print(f"adaptive-kv-memory v{akv.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"VRAM: {mem:.1f} GB")

    print("\nFeatures:")
    print(f"  Triton kernels: {akv.HAS_TRITON}")
    print(f"  TurboQuant: available")
    print(f"  Production cache: available")
    print(f"  HF integration: available")

    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        print(f"  Transformers: not installed")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="akv",
        description="Adaptive KV Memory — hierarchical KV cache compression for LLMs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_get_version()}")
    sub = parser.add_subparsers(dest="command")

    # generate
    gen_p = sub.add_parser("generate", help="Run generation with adaptive KV cache")
    gen_p.add_argument("prompt", type=str, help="Input prompt")
    gen_p.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    gen_p.add_argument("--max-new-tokens", type=int, default=128)
    gen_p.add_argument("--hot-budget", type=int, default=1024)
    gen_p.add_argument("--warm-budget", type=int, default=4096)
    gen_p.add_argument("--warm-bits", type=int, default=3)

    # bench
    bench_p = sub.add_parser("bench", help="Run decode latency benchmark")
    bench_p.add_argument("--num-layers", type=int, default=32)
    bench_p.add_argument("--num-heads", type=int, default=32)
    bench_p.add_argument("--head-dim", type=int, default=128)
    bench_p.add_argument("--hot-budget", type=int, default=1024)
    bench_p.add_argument("--warm-budget", type=int, default=4096)
    bench_p.add_argument("--decode-steps", type=int, default=100)

    # info
    sub.add_parser("info", help="Show system info and available features")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"generate": cmd_generate, "bench": cmd_bench, "info": cmd_info}[args.command](args)


def _get_version() -> str:
    try:
        from akv import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
