"""
ContextForge Base LLM Inference CLI.
====================================
Runs baseline deterministic inference with Qwen/Qwen2.5-1.5B-Instruct without
retrieval, corpus context, or LoRA adapters.

Usage:
    python scripts/run_base.py "What is broken access control?"
    python scripts/run_base.py "What is the Govern function in NIST CSF 2.0?" --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time

from contextforge.pipelines.base import BasePipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_base",
        description="Run ContextForge baseline LLM inference.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Cybersecurity question to answer.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cpu, cuda (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Weight dtype: auto, float32, bfloat16, float16 (default: auto).",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    try:
        t_load_start = time.perf_counter()
        print(f"Loading BasePipeline with model '{args.model_id}'...")
        pipeline = BasePipeline(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        # Pre-load to distinguish cold load from generation latency
        _ = pipeline.model_loader.model
        load_time_sec = time.perf_counter() - t_load_start
        print(f"Model ready in {load_time_sec:.2f}s (Device: {pipeline.model_loader.device}, Dtype: {pipeline.model_loader.dtype}).")

        print(f"\n========================================================")
        print(f"Question: {args.question}")
        print(f"========================================================")

        result = pipeline.generate(args.question)

        print(f"Pipeline:       {result.pipeline}")
        print(f"Model:          {result.model_id}")
        print(f"Device:         {result.metadata.get('device')}")
        print(f"Latency:        {result.latency_ms:.1f} ms")
        print(f"Input tokens:   {result.metadata.get('input_token_count')}")
        print(f"Output tokens:  {result.metadata.get('output_token_count')}")
        print(f"Sources:        {len(result.sources)} (retrieval inactive in base mode)")
        print(f"========================================================\n")
        print("Answer:\n")
        print(result.answer)
        print("\n========================================================")
        return 0

    except Exception as e:
        print(f"\nError running base inference: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
