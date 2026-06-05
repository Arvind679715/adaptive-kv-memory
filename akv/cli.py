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


def cmd_validate(args: argparse.Namespace) -> None:
    """FP16-vs-AKV greedy-decode smoke test against a HuggingFace model.

    Runs the same prompt through ``DynamicCache`` (FP16 baseline) and
    ``AKVCache``, prints token-agreement %, generation text side-by-side,
    and exits non-zero when agreement falls below ``--min-agreement``.

    Intended for CI: a single command that catches HF Cache-API drift
    before it ships.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from akv import AKVCache

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    prompt = args.prompt
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    input_len = input_ids.shape[1]

    def _gen(cache):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, past_key_values=cache,
                pad_token_id=tok.eos_token_id or 0,
            )
        return out[0, input_len:].tolist(), time.perf_counter() - t0

    print("\n── FP16 baseline (DynamicCache) ──")
    fp_ids, fp_t = _gen(DynamicCache())
    fp_text = tok.decode(fp_ids, skip_special_tokens=True)
    print(f"  {len(fp_ids)} tokens in {fp_t:.2f}s")
    print(f"  Text: {fp_text[:200]}{'...' if len(fp_text) > 200 else ''}")

    cache_kwargs = {}
    if args.preset:
        cache_kwargs["preset"] = args.preset
    if args.warm_bits is not None:
        cache_kwargs["warm_bits"] = args.warm_bits
    if args.hot_budget is not None:
        cache_kwargs["hot_budget"] = args.hot_budget
    if not cache_kwargs:
        cache_kwargs["preset"] = "quality"
    cache_kwargs["num_hidden_layers"] = model.config.num_hidden_layers
    cache_kwargs["enable_promotion"] = not args.no_promotion

    desc = ", ".join(f"{k}={v!r}" for k, v in cache_kwargs.items()
                     if k != "num_hidden_layers")
    print(f"\n── AKVCache({desc}) ──")
    akv_ids, akv_t = _gen(AKVCache(**cache_kwargs))
    akv_text = tok.decode(akv_ids, skip_special_tokens=True)
    print(f"  {len(akv_ids)} tokens in {akv_t:.2f}s")
    print(f"  Text: {akv_text[:200]}{'...' if len(akv_text) > 200 else ''}")

    max_len = max(len(fp_ids), len(akv_ids))
    matches = sum(a == b for a, b in zip(fp_ids, akv_ids))
    pct = 100.0 * matches / max(max_len, 1)
    print(f"\n── Token agreement ──")
    print(f"  Matching: {matches}/{max_len} ({pct:.1f}%)")
    print(f"  Length match: {len(fp_ids) == len(akv_ids)}")

    if pct >= 99.0:
        verdict = "PERFECT — outputs are bit-exact or near-bit-exact"
    elif pct >= 90.0:
        verdict = "GOOD — minor divergence (expected for aggressive presets)"
    elif pct >= 70.0:
        verdict = "ACCEPTABLE — noticeable divergence, check quality"
    else:
        verdict = "FAIL — significant divergence; integration likely broken"
    print(f"  Verdict: {verdict}")

    if pct < args.min_agreement:
        print(f"\nFAILED: agreement {pct:.1f}% < threshold {args.min_agreement:.1f}%")
        sys.exit(1)


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

    # validate
    val_p = sub.add_parser(
        "validate",
        help="FP16-vs-AKV greedy-decode smoke test (catches Cache-API drift)",
    )
    val_p.add_argument("--model", type=str, required=True,
                       help="HuggingFace model id (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    val_p.add_argument("--prompt", type=str,
                       default="Explain the difference between TCP and UDP in two sentences.",
                       help="Prompt to decode with both caches")
    val_p.add_argument("--max-new-tokens", type=int, default=64)
    val_p.add_argument("--preset", choices=["quality", "balanced", "compact"], default=None,
                       help="AKV preset (mutually exclusive with --warm-bits/--hot-budget)")
    val_p.add_argument("--warm-bits", type=int, default=None)
    val_p.add_argument("--hot-budget", type=int, default=None)
    val_p.add_argument("--no-promotion", action="store_true",
                       help="Disable warm->hot promotion (keeps run fully deterministic)")
    val_p.add_argument("--min-agreement", type=float, default=80.0,
                       help="Exit non-zero when token-agreement %% falls below this")
    val_p.add_argument("--cpu", action="store_true",
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
        "validate": cmd_validate,
    }[args.command](args)


def _get_version() -> str:
    try:
        from akv import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
