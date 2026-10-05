"""
Focused Phase 5 RAG Evaluation Script (Ponytail / Minimal Work).
===============================================================
Executes:
1. EXP-003: Controlled Top-K Engineering Experiment (K=3, 5, 8) on medium index.
   - Retrieval and context metrics measured across 3 domain queries.
   - 1 representative generation per K (max 3 generations).
   - Moderate default selection: top_k = 5.
2. Real RAG qualitative verification (selected K=5):
   - OWASP Top 10:2025 taxonomy test
   - NIST CSF 2.0 Govern function test
   - CISA CPG 2.0 evidence test
   - Unsupported domain refusal test (chocolate cake recipe)
3. Persists telemetry and results to artifacts/rag_phase5_results.json.

Loads the shared Qwen2.5-1.5B-Instruct model once to avoid repeated model loading on CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from contextforge.llm.base_model import BaseModelLoader
from contextforge.pipelines.rag import RagPipeline

TOPK_EXPERIMENT_QUESTIONS = [
    "What is broken access control?",
    "What is the Govern function in NIST CSF 2.0?",
    "What do the CISA Cross-Sector Cybersecurity Performance Goals provide?",
]

REPRESENTATIVE_TOPK_QUESTION = "What is broken access control?"

REAL_VERIFICATION_QUESTIONS = [
    {
        "id": "owasp_taxonomy",
        "category": "OWASP 2025 Taxonomy",
        "question": "What are the official ten vulnerability categories defined in OWASP Top 10:2025?",
    },
    {
        "id": "nist_govern",
        "category": "NIST CSF 2.0 Govern Function",
        "question": "What is the Govern function in NIST CSF 2.0?",
    },
    {
        "id": "cisa_cpg",
        "category": "CISA CPG 2.0 Goals",
        "question": "What do the CISA Cross-Sector Cybersecurity Performance Goals provide?",
    },
    {
        "id": "unsupported_domain",
        "category": "Unsupported Domain Refusal",
        "question": "What is the best recipe for chocolate cake?",
    },
]


def run_evaluation() -> None:
    print("=" * 70)
    print("ContextForge Phase 5: RAG Pipeline Minimal Evaluation (Ponytail Mode)")
    print("=" * 70)

    # 1. Load shared model once
    print("\n[1/3] Loading shared Qwen2.5-1.5B-Instruct model...")
    t0 = time.perf_counter()
    shared_loader = BaseModelLoader(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    _ = shared_loader.model  # Trigger load
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s on {shared_loader.device} ({shared_loader.dtype}).\n")

    # 2. Run Top-K Engineering Experiment (K=3, 5, 8)
    print("[2/3] Executing EXP-003: Top-K Engineering Comparison (K=3, 5, 8)...")
    topk_results: dict[str, Any] = {"k_values": [3, 5, 8], "runs": {}, "selected_k": 5}

    for k in [3, 5, 8]:
        print(f"\n--- Testing K={k} ---")
        pipeline_k = RagPipeline(
            model_loader=shared_loader,
            chunk_profile="medium",
            top_k=k,
            max_new_tokens=180,
        )

        retrieval_summaries = []
        for q in TOPK_EXPERIMENT_QUESTIONS:
            ret_t0 = time.perf_counter()
            chunks = pipeline_k.retriever.retrieve(q, top_k=k)
            ret_ms = (time.perf_counter() - ret_t0) * 1000.0
            formatted = pipeline_k.formatter.format(chunks)

            sources_meta = [
                {
                    "source_label": c.source_label,
                    "doc_id": c.document_id,
                    "score": round(c.score, 4),
                }
                for c in chunks
            ]
            retrieval_summaries.append({
                "question": q,
                "retrieved_count": len(chunks),
                "included_count": formatted.included_count,
                "context_token_count": formatted.context_token_count,
                "retrieval_latency_ms": round(ret_ms, 2),
                "sources": sources_meta,
            })
            print(f"  [Ret: {ret_ms:.1f}ms] '{q}' -> {len(chunks)} chunks, {formatted.context_token_count} context tokens")

        # 1 representative generation per K
        print(f"  Generating representative answer for '{REPRESENTATIVE_TOPK_QUESTION}' (K={k})...")
        gen_res = pipeline_k.generate(REPRESENTATIVE_TOPK_QUESTION)
        gen_meta = gen_res.metadata

        topk_results["runs"][f"k_{k}"] = {
            "retrieval_metrics": retrieval_summaries,
            "avg_context_tokens": sum(r["context_token_count"] for r in retrieval_summaries) / len(retrieval_summaries),
            "avg_retrieval_ms": sum(r["retrieval_latency_ms"] for r in retrieval_summaries) / len(retrieval_summaries),
            "representative_generation": {
                "question": REPRESENTATIVE_TOPK_QUESTION,
                "context_token_count": gen_meta["context_token_count"],
                "output_token_count": gen_meta["output_token_count"],
                "generation_latency_ms": gen_meta["generation_latency_ms"],
                "total_latency_ms": gen_res.latency_ms,
                "citations_used": gen_meta["citation_labels_used"],
                "invalid_citations": gen_meta["invalid_citations"],
                "answer_snippet": gen_res.answer[:200] + "...",
            },
        }
        pipeline_k.close()

    print("\nTop-K Comparison Summary:")
    for k in [3, 5, 8]:
        run = topk_results["runs"][f"k_{k}"]
        rep = run["representative_generation"]
        print(
            f"  K={k}: Avg Context Tokens={run['avg_context_tokens']:.1f}, "
            f"Avg Ret Latency={run['avg_retrieval_ms']:.1f}ms, "
            f"Gen Latency={rep['generation_latency_ms']:.1f}ms, "
            f"Citations={rep['citations_used']}"
        )
    print("Selected K=5 as default engineering decision: optimal balance of context coverage and token budget.")

    # 3. Real RAG Qualitative Verification (Selected K=5)
    print("\n[3/3] Executing Real RAG Qualitative Verification (Selected K=5)...")
    rag_default = RagPipeline(
        model_loader=shared_loader,
        chunk_profile="medium",
        top_k=5,
        max_new_tokens=256,
    )

    verification_results = []
    for item in REAL_VERIFICATION_QUESTIONS:
        qid = item["id"]
        cat = item["category"]
        q = item["question"]

        print(f"\n[{qid}] ({cat})")
        print(f"Question: {q}")
        res = rag_default.generate(q)
        meta = res.metadata

        sources_summary = []
        for src in res.sources[:meta["included_count"]]:
            heading = src.metadata.get("section_heading") or f"Page {src.metadata.get('page_start', '')}"
            sources_summary.append({
                "source_label": src.source_label,
                "document_id": src.document_id,
                "reference": heading,
                "score": round(src.score, 4),
            })

        print(f"Latency: {res.latency_ms:.1f}ms (Ret: {meta['retrieval_latency_ms']:.1f}ms, Gen: {meta['generation_latency_ms']:.1f}ms)")
        print(f"Context tokens: {meta['context_token_count']}, Output tokens: {meta['output_token_count']}")
        print(f"Citations: {meta['citation_labels_used']} (Invalid: {meta['invalid_citations']})")
        print("Answer:\n" + res.answer)

        verification_results.append({
            "id": qid,
            "category": cat,
            "question": q,
            "answer": res.answer,
            "latency_ms": res.latency_ms,
            "retrieval_latency_ms": meta["retrieval_latency_ms"],
            "generation_latency_ms": meta["generation_latency_ms"],
            "context_token_count": meta["context_token_count"],
            "input_token_count": meta["input_token_count"],
            "output_token_count": meta["output_token_count"],
            "citation_labels_used": meta["citation_labels_used"],
            "invalid_citations": meta["invalid_citations"],
            "sources": sources_summary,
        })

    rag_default.close()

    # Save output artifacts
    output_path = Path("artifacts/rag_phase5_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": shared_loader.model_id,
        "device": str(shared_loader.device),
        "dtype": str(shared_loader.dtype),
        "chunk_profile": "medium",
        "top_k_experiment": topk_results,
        "real_verification": verification_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete! Results persisted to '{output_path}'.")


if __name__ == "__main__":
    run_evaluation()
