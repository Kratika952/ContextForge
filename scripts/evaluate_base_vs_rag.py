"""
Phase 6 Evaluation Runner: BasePipeline vs RagPipeline.
=======================================================
Executes the Phase 6 development evaluation comparing Base LLM vs RAG on identical
questions, identical deterministic generation settings, and identical hardware.

- Loads Qwen2.5-1.5B-Instruct model ONCE into memory.
- Evaluates 12 curated questions across OWASP Top 10:2025, NIST CSF 2.0, CISA CPG 2.0, and unsupported queries.
- Computes deterministic, explainable metrics (coverage, latency, tokens, retrieval, citations, refusal).
- Saves raw results to artifacts/evaluation/base_vs_rag_v1.json.
- Outputs a concise comparison report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from contextforge.evaluation.evaluator import (
    aggregate_evaluation_results,
    evaluate_question_pair,
    load_evaluation_dataset,
)
from contextforge.llm.base_model import BaseModelLoader
from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.rag import RagPipeline

DEFAULT_DATASET_PATH = Path("data/qa/evaluation_v1.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/evaluation/base_vs_rag_v1.json")


def run_evaluation(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Execute complete Base vs RAG evaluation run.

    Args:
        dataset_path: Path to evaluation questions JSONL.
        output_path: Path where raw results JSON will be written.

    Returns:
        Dictionary of aggregated evaluation metrics.
    """
    dataset_file = Path(dataset_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ContextForge Phase 6: Base vs RAG Evaluation Harness v1")
    print("=" * 72)
    sys.stdout.flush()

    # 1. Load dataset
    questions = load_evaluation_dataset(dataset_file)
    print(f"Loaded {len(questions)} evaluation questions from {dataset_file}")

    # Count categories
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.expected_source] = counts.get(q.expected_source, 0) + 1
    print(f"Categories: {counts}")
    sys.stdout.flush()

    # 2. Load shared Qwen model once
    print("\nLoading shared Qwen/Qwen2.5-1.5B-Instruct model (single CPU process)...")
    t0 = time.perf_counter()
    shared_loader = BaseModelLoader(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    _ = shared_loader.model  # Trigger load
    load_sec = time.perf_counter() - t0
    print(f"Model loaded in {load_sec:.1f}s on {shared_loader.device} ({shared_loader.dtype}).\n")
    sys.stdout.flush()

    # 3. Instantiate pipelines reusing the shared model loader
    base_pipeline = BasePipeline(
        model_loader=shared_loader,
        max_new_tokens=256,
        prompt_version="base_v1",
    )
    rag_pipeline = RagPipeline(
        model_loader=shared_loader,
        chunk_profile="medium",
        top_k=5,
        max_new_tokens=256,
        prompt_version="rag_v1",
    )

    # 4. Run evaluation loop
    evaluated_pairs: list[dict[str, Any]] = []

    print("-" * 72)
    print(f"{'ID':<10} {'Source':<8} {'Base Cov':<10} {'RAG Cov':<10} {'Base Lat':<10} {'RAG Lat':<10}")
    print("-" * 72)
    sys.stdout.flush()

    for idx, q in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Running question: {q.id} ('{q.question[:45]}...')", flush=True)

        # Base inference
        t_base = time.perf_counter()
        base_res = base_pipeline.generate(q.question)
        base_dur = time.perf_counter() - t_base

        # RAG inference
        t_rag = time.perf_counter()
        rag_res = rag_pipeline.generate(q.question)
        rag_dur = time.perf_counter() - t_rag

        # Evaluate pair
        pair = evaluate_question_pair(q, base_res, rag_res)
        evaluated_pairs.append(pair)

        base_cov = pair["base"]["reference_coverage"]
        rag_cov = pair["rag"]["reference_coverage"]
        base_ms = pair["base"]["latency_ms"]
        rag_ms = pair["rag"]["latency_ms"]

        print(
            f"  -> {q.id:<8} {q.expected_source:<8} {base_cov:<10.2f} {rag_cov:<10.2f} {base_ms:<10.0f}ms {rag_ms:<10.0f}ms",
            flush=True,
        )

    # 5. Clean up pipeline connections
    rag_pipeline.close()

    # 6. Compute aggregates
    summary = aggregate_evaluation_results(evaluated_pairs)

    # 7. Serialize raw and aggregated results
    run_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": "EXP-004",
        "model_id": shared_loader.model_id,
        "device": str(shared_loader.device),
        "dtype": str(shared_loader.dtype),
        "decoding": "deterministic_greedy",
        "dataset_file": str(dataset_file),
        "total_questions": len(questions),
        "summary": summary,
        "items": evaluated_pairs,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY: Base vs RAG (EXP-004)")
    print("=" * 72)
    base_agg = summary["base"]
    rag_agg = summary["rag"]

    print(f"{'Metric':<32} {'Base':<18} {'RAG':<18}")
    print("-" * 72)
    print(f"{'Reference Coverage':<32} {base_agg['avg_reference_coverage']:<18.4f} {rag_agg['avg_reference_coverage']:<18.4f}")
    print(f"{'Average Latency (ms)':<32} {base_agg['avg_latency_ms']:<18.1f} {rag_agg['avg_total_latency_ms']:<18.1f}")
    print(f"{'  - Retrieval Latency (ms)':<32} {'N/A':<18} {rag_agg['avg_retrieval_latency_ms']:<18.1f}")
    print(f"{'  - Generation Latency (ms)':<32} {base_agg['avg_latency_ms']:<18.1f} {rag_agg['avg_generation_latency_ms']:<18.1f}")
    print(f"{'Average Input Tokens':<32} {base_agg['avg_input_tokens']:<18.1f} {rag_agg['avg_input_tokens']:<18.1f}")
    print(f"{'Average Output Tokens':<32} {base_agg['avg_output_tokens']:<18.1f} {rag_agg['avg_output_tokens']:<18.1f}")
    print(f"{'Average Context Tokens':<32} {'0.0':<18} {rag_agg['avg_context_tokens']:<18.1f}")
    print("-" * 72)
    print("RAG-Specific Metrics:")
    print(f"  - Expected Source Retrieval Rate: {rag_agg['expected_source_retrieval_rate'] * 100:.1f}%")
    print(f"  - Citation Frequency:             {rag_agg['citation_frequency'] * 100:.1f}%")
    print(f"  - Citation Validity:              {rag_agg['citation_validity'] * 100:.1f}%")
    print(f"  - Unsupported Refusal Rate:       {rag_agg['unsupported_refusal_rate'] * 100:.1f}%")
    print("-" * 72)
    print(f"Raw results persisted to: {output_file}")
    sys.stdout.flush()

    return run_payload


if __name__ == "__main__":
    run_evaluation()
