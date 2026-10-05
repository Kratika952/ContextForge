"""
ContextForge RAG Inference CLI.
===============================
Runs evidence-grounded question answering using DenseRetriever and Qwen2.5-1.5B-Instruct.

Usage:
    python scripts/run_rag.py "What are the OWASP Top 10:2025 categories?"
    python scripts/run_rag.py "What is the Govern function in NIST CSF 2.0?" -k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contextforge.pipelines.rag import RagPipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_rag",
        description="Run RAG inference using ContextForge knowledge base and Qwen2.5 LLM.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Cybersecurity question to answer.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Chunking profile to query (default: medium).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation output tokens (default: 256).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=1500,
        help="Maximum context token budget for evidence passages (default: 1500).",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    print(f"\n[ContextForge RAG Pipeline]")
    print(f"Loading retriever and LLM...")

    try:
        pipeline = RagPipeline(
            top_k=args.top_k,
            chunk_profile=args.profile,
            max_new_tokens=args.max_tokens,
            max_context_tokens=args.max_context_tokens,
        )
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}", file=sys.stderr)
        return 1

    try:
        result = pipeline.generate(args.question)
    except Exception as e:
        print(f"Error during RAG generation: {e}", file=sys.stderr)
        return 1

    meta = result.metadata

    print(f"\n========================================================")
    print(f"Pipeline: {result.pipeline}")
    print(f"Model: {result.model_id}")
    print(f"Chunk profile: {meta.get('chunk_profile', args.profile)}")
    print(f"Retrieved: {meta.get('retrieved_count', len(result.sources))} (Included: {meta.get('included_count', 0)})")
    print(f"Context tokens: {meta.get('context_token_count', 0)}")
    print(f"Input tokens: {meta.get('input_token_count', 0)} | Output tokens: {meta.get('output_token_count', 0)}")
    print(f"Retrieval latency: {meta.get('retrieval_latency_ms', 0.0):.1f} ms")
    print(f"Generation latency: {meta.get('generation_latency_ms', 0.0):.1f} ms")
    print(f"Total latency: {result.latency_ms:.1f} ms")
    if meta.get("citation_labels_used"):
        print(f"Citations used: {meta['citation_labels_used']}")
    if meta.get("invalid_citations"):
        print(f"Invalid citations flagged: {meta['invalid_citations']}")
    print(f"========================================================\n")

    print("Answer:")
    print(result.answer)
    print()

    print("Sources:")
    if not result.sources:
        print("  (None retrieved)")
    else:
        for i, src in enumerate(result.sources[:meta.get('included_count', len(result.sources))], start=1):
            source_info = f"[{i}] {src.source_label} (Doc: {src.document_id}, Score: {src.score:.4f})"
            if src.metadata.get("section_heading"):
                source_info += f" - Section: {src.metadata['section_heading']}"
            elif src.metadata.get("page_start"):
                p_end = src.metadata.get("page_end", src.metadata["page_start"])
                source_info += f" - Page: {src.metadata['page_start']}" + (f"-{p_end}" if p_end != src.metadata["page_start"] else "")
            print(f"  {source_info}")
    print()

    pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
