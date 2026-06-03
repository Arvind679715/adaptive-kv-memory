"""AKV command-line interface for compressed KV cache generation and benchmarking."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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
    print(f"  NormQuant: available")
    print(f"  Production cache: available")
    print(f"  HF integration: available")

    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        print(f"  Transformers: not installed")


def cmd_adapters(args: argparse.Namespace) -> None:
    """List supported model architectures."""
    from akv.adapters import list_adapters
    print(f"{'family':<35} {'model_type':<18} {'preset':<10} status")
    print("-" * 80)
    for spec in list_adapters():
        status = "ok" if spec.supported else "UNSUPPORTED"
        print(f"{spec.family:<35} {spec.model_type:<18} {spec.default_preset:<10} {status}")
        if args.verbose and spec.notes:
            for line in spec.notes.split("\n"):
                print(f"    {line}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run a calibration pass and save a config the user can re-load."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from akv.calibration import calibrate_model

    print(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    print(f"Calibrating on {device} (target {args.target_bits}b avg) ...")
    sample_texts = None
    if args.calibration_file:
        sample_texts = [Path(args.calibration_file).read_text()]

    report = calibrate_model(
        model, tok,
        sample_texts=sample_texts,
        max_length=args.max_length,
        max_layers_to_probe=args.probe_layers,
        target_average_bits=args.target_bits,
        device=device,
    )
    print("")
    print(report.summary)
    print("")
    print(f"Calibration took {report.calibration_seconds:.1f}s")

    out = Path(args.output)
    report.save(out)
    print(f"Saved calibration to: {out}")


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

    # adapters
    adapters_p = sub.add_parser("adapters", help="List supported model architectures")
    adapters_p.add_argument("-v", "--verbose", action="store_true",
                            help="Show notes / caveats for each adapter")

    # calibrate
    cal_p = sub.add_parser("calibrate",
                           help="Run calibration on a model and save a JSON config")
    cal_p.add_argument("--model", type=str, required=True,
                       help="HuggingFace model id or local path")
    cal_p.add_argument("--output", "-o", type=str, default="akv_calibration.json")
    cal_p.add_argument("--calibration-file", type=str, default=None,
                       help="Optional path to a text file used as calibration data")
    cal_p.add_argument("--max-length", type=int, default=1024)
    cal_p.add_argument("--probe-layers", type=int, default=8,
                       help="Number of layers to probe (rest extrapolated)")
    cal_p.add_argument("--target-bits", type=float, default=3.0,
                       help="Target average bits per head (2.0 - 4.0)")
    cal_p.add_argument("--cpu", action="store_true",
                       help="Force CPU even if CUDA is available")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {
        "generate": cmd_generate,
        "bench": cmd_bench,
        "info": cmd_info,
        "adapters": cmd_adapters,
        "calibrate": cmd_calibrate,
    }[args.command](args)


def _get_version() -> str:
    try:
        from akv import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
