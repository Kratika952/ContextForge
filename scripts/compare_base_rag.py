"""
ContextForge Base vs RAG Single-Process Developer Comparison Script.
===================================================================
Runs a question through BasePipeline and RagPipeline in a single process
sharing the same cached Qwen model instance to avoid expensive redundant model loads.

Usage:
    python scripts/compare_base_rag.py "What are the official ten vulnerability categories defined in OWASP Top 10:2025?"
    python scripts/compare_base_rag.py "What is broken access control?"
"""

from __future__ import annotations

import argparse
import sys
import time

from contextforge.llm.base_model import BaseModelLoader
from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.rag import RagPipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_base_rag",
        description="Compare Base vs RAG pipelines on the same question in one process.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question to evaluate across Base and RAG.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks for RAG (default: 5).",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Chunking profile for RAG index (default: medium).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation output tokens (default: 256).",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    print("\n========================================================")
    print(f"QUESTION: '{args.question}'")
    print("========================================================\n")

    print("[1/3] Loading shared Qwen2.5-1.5B-Instruct model...")
    t_load_0 = time.perf_counter()
    shared_loader = BaseModelLoader(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    _ = shared_loader.model  # Trigger load
    t_load = time.perf_counter() - t_load_0
    print(f"Shared model loaded in {t_load:.1f}s on {shared_loader.device} ({shared_loader.dtype}).\n")

    # 1. Base Pipeline
    print("[2/3] Executing Base Pipeline (unmodified Qwen)...")
    base_pipeline = BasePipeline(
        model_loader=shared_loader,
        max_new_tokens=args.max_tokens,
    )
    base_res = base_pipeline.generate(args.question)

    # 2. RAG Pipeline
    print("[3/3] Executing RAG Pipeline (DenseRetriever + Qwen)...")
    rag_pipeline = RagPipeline(
        model_loader=shared_loader,
        top_k=args.top_k,
        chunk_profile=args.profile,
        max_new_tokens=args.max_tokens,
    )
    rag_res = rag_pipeline.generate(args.question)

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n--- BASE PIPELINE ---")
    print(f"Latency: {base_res.latency_ms:.1f} ms | Output tokens: {base_res.metadata.get('output_token_count', 0)}")
    print(f"Sources: {len(base_res.sources)}")
    print("Answer:")
    print(base_res.answer)

    print("\n--- RAG PIPELINE ---")
    rag_meta = rag_res.metadata
    print(
        f"Retrieval latency: {rag_meta.get('retrieval_latency_ms', 0.0):.1f} ms | "
        f"Generation latency: {rag_meta.get('generation_latency_ms', 0.0):.1f} ms | "
        f"Total latency: {rag_res.latency_ms:.1f} ms"
    )
    print(
        f"Retrieved: {rag_meta.get('retrieved_count', 0)} | "
        f"Included: {rag_meta.get('included_count', 0)} | "
        f"Context tokens: {rag_meta.get('context_token_count', 0)}"
    )
    print(f"Citations used: {rag_meta.get('citation_labels_used', [])}")
    if rag_meta.get("invalid_citations"):
        print(f"Invalid citations: {rag_meta['invalid_citations']}")
    print("Answer:")
    print(rag_res.answer)

    print("\nRetrieved Sources:")
    for i, src in enumerate(rag_res.sources[:rag_meta.get('included_count', len(rag_res.sources))], start=1):
        heading = src.metadata.get("section_heading") or f"Page {src.metadata.get('page_start', '')}"
        print(f"  [{i}] {src.source_label} ({src.document_id}) - {heading} (Score: {src.score:.4f})")

    print("\n" + "=" * 70)

    rag_pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
