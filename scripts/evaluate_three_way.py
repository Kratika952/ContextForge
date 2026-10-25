"""ContextForge — Phase 9 Final Three-Way Comparative Evaluation Runner.
=====================================================================
Executes the fair comparative evaluation across:
A. Base LLM (Qwen/Qwen2.5-1.5B-Instruct)
B. RAG (Same base LLM + BGE embeddings + FAISS)
C. Fine-Tuned (Same base LLM + QLoRA adapter)

Runs sequentially to respect the 4 GB GPU VRAM limit on the RTX 2050.
Persists raw predictions before computing aggregate evaluation metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from contextforge.common.types import AnswerResult
from contextforge.evaluation.evaluator import (
    aggregate_phase9_results,
    evaluate_phase9_triplet,
    load_phase9_test_dataset,
)
from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.finetuned import FineTunedPipeline
from contextforge.pipelines.rag import RagPipeline
from contextforge.retrieval.embedder import BGEEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("contextforge.phase9")

DEFAULT_DATASET_PATH = Path("data/qa/phase9_final_test.jsonl")
DEFAULT_RAW_OUTPUT_PATH = Path("artifacts/evaluation/phase9_raw_predictions.json")
DEFAULT_METRICS_OUTPUT_PATH = Path("artifacts/evaluation/phase9_metrics.json")
DEFAULT_ADAPTER_PATH = Path("artifacts/finetuning/phase8_pilot")


def run_three_way_evaluation(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    raw_output_path: Path | str = DEFAULT_RAW_OUTPUT_PATH,
    metrics_output_path: Path | str = DEFAULT_METRICS_OUTPUT_PATH,
    adapter_path: Path | str = DEFAULT_ADAPTER_PATH,
    device: str = "auto",
) -> dict[str, Any]:
    """Execute complete three-way evaluation across Base, RAG, and Fine-Tuned systems."""
    dataset_file = Path(dataset_path)
    raw_file = Path(raw_output_path)
    metrics_file = Path(metrics_output_path)
    adapter_dir = Path(adapter_path)

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 9 — FINAL COMPARATIVE EVALUATION (BASE vs RAG vs FINE-TUNED)")
    print("=" * 80)

    # 1. Load held-out test questions
    questions = load_phase9_test_dataset(dataset_file)
    logger.info("Loaded %d held-out test questions from '%s'", len(questions), dataset_file)

    def cleanup_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    intermediate_store: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_file": str(dataset_file),
            "num_test_records": len(questions),
            "device": device,
        },
        "base_results": {},
        "rag_results": {},
        "finetuned_results": {},
        "peak_memory_mb": {},
    }

    # ── Phase 1: Base LLM ───────────────────────────────────────────────────
    cleanup_memory()
    logger.info("Initializing BasePipeline on %s...", device)
    base_pipe = BasePipeline(device=device, prompt_version="base_v1", max_new_tokens=256)
    logger.info("Running BasePipeline across %d items...", len(questions))

    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        res = base_pipe.generate(q.question)
        lat = (time.perf_counter() - t0) * 1000.0
        intermediate_store["base_results"][q.id] = {
            "answer": res.answer,
            "latency_ms": lat,
            "metadata": res.metadata,
        }
        print(f"  [Base {idx}/{len(questions)}] {q.id} ({lat:.1f}ms)", flush=True)

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        peak_b = torch.cuda.max_memory_allocated() / (1024 * 1024)
        intermediate_store["peak_memory_mb"]["base"] = round(peak_b, 1)
        logger.info("BasePipeline peak GPU memory: %.1f MB", peak_b)

    del base_pipe
    cleanup_memory()
    logger.info("BasePipeline unloaded and memory cleared.")

    # Save incremental state
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(intermediate_store, f, indent=2)

    # ── Phase 2: RAG Pipeline ───────────────────────────────────────────────
    cleanup_memory()
    logger.info("Initializing RagPipeline on %s (chunk_profile='medium', top_k=5)...", device)
    rag_pipe = RagPipeline(
        device=device,
        chunk_profile="medium",
        top_k=5,
        max_context_tokens=1500,
        prompt_version="rag_v1",
        max_new_tokens=256,
    )
    logger.info("Running RagPipeline across %d items...", len(questions))

    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        res = rag_pipe.generate(q.question)
        lat = (time.perf_counter() - t0) * 1000.0
        intermediate_store["rag_results"][q.id] = {
            "answer": res.answer,
            "latency_ms": lat,
            "metadata": res.metadata,
            "sources": [
                {
                    "document_id": s.document_id,
                    "chunk_id": s.chunk_id,
                    "score": s.score,
                    "source_label": s.source_label,
                    "metadata": s.metadata,
                }
                for s in res.sources
            ],
        }
        print(f"  [RAG {idx}/{len(questions)}] {q.id} ({lat:.1f}ms)", flush=True)

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        peak_r = torch.cuda.max_memory_allocated() / (1024 * 1024)
        intermediate_store["peak_memory_mb"]["rag"] = round(peak_r, 1)
        logger.info("RagPipeline peak GPU memory: %.1f MB", peak_r)

    rag_pipe.close()
    del rag_pipe
    cleanup_memory()
    logger.info("RagPipeline unloaded and memory cleared.")

    # Save incremental state
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(intermediate_store, f, indent=2)

    # ── Phase 3: Fine-Tuned Pipeline ────────────────────────────────────────
    cleanup_memory()
    logger.info("Initializing FineTunedPipeline with adapter '%s' on %s...", adapter_dir, device)
    ft_pipe = FineTunedPipeline(
        adapter_path=adapter_dir,
        device=device,
        prompt_version="base_v1",
        max_new_tokens=256,
    )
    logger.info("Running FineTunedPipeline across %d items...", len(questions))

    for idx, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        res = ft_pipe.generate(q.question)
        lat = (time.perf_counter() - t0) * 1000.0
        intermediate_store["finetuned_results"][q.id] = {
            "answer": res.answer,
            "latency_ms": lat,
            "metadata": res.metadata,
        }
        print(f"  [Fine-Tuned {idx}/{len(questions)}] {q.id} ({lat:.1f}ms)", flush=True)

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        peak_ft = torch.cuda.max_memory_allocated() / (1024 * 1024)
        intermediate_store["peak_memory_mb"]["finetuned"] = round(peak_ft, 1)
        logger.info("FineTunedPipeline peak GPU memory: %.1f MB", peak_ft)

    del ft_pipe
    cleanup_memory()
    logger.info("FineTunedPipeline unloaded and memory cleared.")

    # Save complete raw predictions artifact
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(intermediate_store, f, indent=2)
    logger.info("Raw predictions successfully saved to '%s'", raw_file)

    # ── Phase 4: Compute Metrics ────────────────────────────────────────────
    logger.info("Loading BGE embedder for semantic similarity evaluation...")
    cleanup_memory()
    embedder = BGEEmbedder(device="cpu")

    from contextforge.common.types import RetrievedChunk

    item_evaluations: list[dict[str, Any]] = []

    for q in questions:
        b_data = intermediate_store["base_results"][q.id]
        r_data = intermediate_store["rag_results"][q.id]
        ft_data = intermediate_store["finetuned_results"][q.id]

        base_res = AnswerResult(
            question=q.question,
            pipeline="base",
            answer=b_data["answer"],
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            latency_ms=b_data["latency_ms"],
            metadata=b_data["metadata"],
        )
        rag_sources = [
            RetrievedChunk(
                chunk_id=s["chunk_id"],
                document_id=s["document_id"],
                text="",
                score=s["score"],
                rank=idx + 1,
                source_label=s["source_label"],
                metadata=s.get("metadata", {}),
            )
            for idx, s in enumerate(r_data.get("sources", []))
        ]
        rag_res = AnswerResult(
            question=q.question,
            pipeline="rag",
            answer=r_data["answer"],
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            sources=rag_sources,
            latency_ms=r_data["latency_ms"],
            metadata=r_data["metadata"],
        )
        ft_res = AnswerResult(
            question=q.question,
            pipeline="finetuned",
            answer=ft_data["answer"],
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            latency_ms=ft_data["latency_ms"],
            metadata=ft_data["metadata"],
        )


        item_eval = evaluate_phase9_triplet(
            question=q,
            base_result=base_res,
            rag_result=rag_res,
            finetuned_result=ft_res,
            embedder=embedder,
        )
        item_evaluations.append(item_eval)

    aggregate_summary = aggregate_phase9_results(item_evaluations)
    aggregate_summary["peak_gpu_memory_mb"] = intermediate_store["peak_memory_mb"]

    final_payload = {
        "metadata": intermediate_store["metadata"],
        "summary": aggregate_summary,
        "items": item_evaluations,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    logger.info("Aggregated metrics successfully saved to '%s'", metrics_file)

    # Output human-readable summary table
    print("\n" + "=" * 80)
    print("PHASE 9 — EVALUATION SUMMARY (BASE vs RAG vs FINE-TUNED)")
    print("=" * 80)
    b_agg = aggregate_summary["base"]
    r_agg = aggregate_summary["rag"]
    ft_agg = aggregate_summary["finetuned"]

    print(f"{'Metric':<32} {'Base':<16} {'RAG':<16} {'Fine-Tuned':<16}")
    print("-" * 80)
    print(f"{'Token Precision (mean)':<32} {b_agg['token_precision_mean']:<16.4f} {r_agg['token_precision_mean']:<16.4f} {ft_agg['token_precision_mean']:<16.4f}")
    print(f"{'Token Recall (mean)':<32} {b_agg['token_recall_mean']:<16.4f} {r_agg['token_recall_mean']:<16.4f} {ft_agg['token_recall_mean']:<16.4f}")
    print(f"{'Token F1 (mean)':<32} {b_agg['token_f1_mean']:<16.4f} {r_agg['token_f1_mean']:<16.4f} {ft_agg['token_f1_mean']:<16.4f}")
    print(f"{'Semantic Similarity (mean)':<32} {b_agg['semantic_similarity_mean']:<16.4f} {r_agg['semantic_similarity_mean']:<16.4f} {ft_agg['semantic_similarity_mean']:<16.4f}")
    print(f"{'Latency Mean (ms)':<32} {b_agg['latency_ms_mean']:<16.1f} {r_agg['total_latency_ms_mean']:<16.1f} {ft_agg['latency_ms_mean']:<16.1f}")
    print(f"{'Latency Median (ms)':<32} {b_agg['latency_ms_median']:<16.1f} {r_agg['total_latency_ms_median']:<16.1f} {ft_agg['latency_ms_median']:<16.1f}")
    print(f"{'Input Tokens (mean)':<32} {b_agg['input_tokens_mean']:<16.1f} {r_agg['input_tokens_mean']:<16.1f} {ft_agg['input_tokens_mean']:<16.1f}")
    print(f"{'Output Tokens (mean)':<32} {b_agg['output_tokens_mean']:<16.1f} {r_agg['output_tokens_mean']:<16.1f} {ft_agg['output_tokens_mean']:<16.1f}")
    print(f"{'Context Tokens (mean)':<32} {'0.0':<16} {r_agg['context_tokens_mean']:<16.1f} {'0.0':<16}")
    print("-" * 80)
    print("RAG Retrieval & Citation Metrics:")
    print(f"  - Document Hit@1:     {r_agg['document_hit_at_1'] * 100:.1f}%")
    print(f"  - Document Hit@3:     {r_agg['document_hit_at_3'] * 100:.1f}%")
    print(f"  - Document Hit@5:     {r_agg['document_hit_at_5'] * 100:.1f}%")
    print(f"  - Document MRR:       {r_agg['document_mrr']:.4f}")
    print(f"  - Citation Presence:  {r_agg['citation_presence_rate'] * 100:.1f}%")
    print(f"  - Citation Validity:  {r_agg['citation_validity'] * 100:.1f}%")
    if intermediate_store["peak_memory_mb"]:
        print("-" * 80)
        print("Peak GPU Memory (MB):")
        for k, v in intermediate_store["peak_memory_mb"].items():
            print(f"  - {k:<12}: {v:.1f} MB")
    print("=" * 80)

    return final_payload


def main() -> None:
    """CLI entrypoint for Phase 9 three-way evaluation."""
    parser = argparse.ArgumentParser(description="ContextForge Phase 9 Three-Way Evaluation")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to Phase 9 test JSONL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--output-raw", type=str, default=str(DEFAULT_RAW_OUTPUT_PATH), help="Path for raw predictions JSON")
    parser.add_argument("--output-metrics", type=str, default=str(DEFAULT_METRICS_OUTPUT_PATH), help="Path for aggregated metrics JSON")
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH), help="Path to trained LoRA adapter")

    args = parser.parse_args()
    run_three_way_evaluation(
        dataset_path=args.dataset,
        raw_output_path=args.output_raw,
        metrics_output_path=args.output_metrics,
        adapter_path=args.adapter_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
