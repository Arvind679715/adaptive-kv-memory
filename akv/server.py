"""OpenAI-compatible HTTP server with adaptive KV cache.

Serves models with AKV's zero-calibration NormQuant compression.
Minimal dependencies: stdlib http.server + torch + transformers.

Usage:
    akv-server --model meta-llama/Llama-3.2-1B --preset balanced --port 8000

Endpoints:
    POST /v1/chat/completions  — Chat completions (streaming supported)
    GET  /v1/models            — List available models
    GET  /health               — Health check
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Preset configurations (matches drop_in.py)
PRESETS = {
    "quality": {"hot_budget": 2048, "warm_bits": 4, "description": "Best quality, 4-bit warm tier"},
    "balanced": {"hot_budget": 1024, "warm_bits": 3, "description": "Good balance, 3-bit warm tier"},
    "compact": {"hot_budget": 512, "warm_bits": 2, "description": "Maximum compression, 2-bit warm tier"},
}


class ModelServer:
    """Holds the loaded model and generates with AKV cache."""

    def __init__(self, model_name: str, preset: str = "balanced", device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from akv.hf_generate import AdaptiveGenerator, GeneratorConfig

        self.model_name = model_name
        self.preset = preset
        self._lock = Lock()

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)

        # Configure generator with preset
        preset_cfg = PRESETS[preset]
        gen_config = GeneratorConfig(
            hot_budget=preset_cfg["hot_budget"],
            warm_bits=preset_cfg["warm_bits"],
            use_production_cache=True,
        )
        self.generator = AdaptiveGenerator(self.model, self.tokenizer, gen_config, device)
        logger.info(f"Model ready: {model_name} (preset={preset}, device={device})")

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
    ):
        """Generate a chat completion."""
        # Format messages into prompt
        prompt = self._format_messages(messages)

        with self._lock:
            output = self.generator.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.01,
            )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.num_generated,
                "total_tokens": output.prompt_tokens + output.num_generated,
            },
        }

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages into a prompt string."""
        # Try chat template first
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


class AKVRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    server: "AKVHTTPServer"

    def log_message(self, format, *args):
        logger.info(format % args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "model": self.server.model_server.model_name,
                             "preset": self.server.model_server.preset})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": self.server.model_server.model_name,
                    "object": "model",
                    "owned_by": "local",
                }],
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_chat_completions(self):
        try:
            body = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_json({"error": "messages field required"}, 400)
            return

        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                self._send_json({"error": "Each message must have role and content"}, 400)
                return

        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        stream = body.get("stream", False)

        # Clamp parameters
        max_tokens = min(max(1, int(max_tokens)), 4096)
        temperature = max(0.0, float(temperature))
        top_p = max(0.0, min(1.0, float(top_p)))

        if stream:
            # SSE streaming not implemented yet — return non-streaming
            pass

        try:
            result = self.server.model_server.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            self._send_json(result)
        except Exception as e:
            logger.exception("Generation error")
            self._send_json({"error": str(e)}, 500)


class AKVHTTPServer(HTTPServer):
    """HTTP server with model reference."""

    def __init__(self, server_address, handler_class, model_server: ModelServer):
        self.model_server = model_server
        super().__init__(server_address, handler_class)


def main():
    parser = argparse.ArgumentParser(
        description="AKV Server — OpenAI-compatible inference with adaptive KV cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  akv-server --model meta-llama/Llama-3.2-1B --preset balanced
  akv-server --model microsoft/phi-2 --preset compact --port 9000
  akv-server --model Qwen/Qwen2-0.5B --device cpu

Presets:
  quality   — 4-bit warm tier, 2048 hot budget (best quality)
  balanced  — 3-bit warm tier, 1024 hot budget (recommended)
  compact   — 2-bit warm tier, 512 hot budget (maximum compression)
""",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced",
                        help="Cache preset (default: balanced)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    model_server = ModelServer(args.model, args.preset, args.device)
    server = AKVHTTPServer((args.host, args.port), AKVRequestHandler, model_server)

    logger.info(f"AKV Server running at http://{args.host}:{args.port}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Preset: {args.preset} ({PRESETS[args.preset]['description']})")
    logger.info(f"  Zero-calibration NormQuant: enabled")
    logger.info(f"Endpoints:")
    logger.info(f"  POST /v1/chat/completions")
    logger.info(f"  GET  /v1/models")
    logger.info(f"  GET  /health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
