"""
ContextForge Retrieval Evaluation Script.
=========================================
Evaluates dense retrieval performance against ground-truth question sets.
Computes Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank), and query latency.

Usage:
    python scripts/evaluate_retrieval.py --profile medium
    python scripts/evaluate_retrieval.py --all-profiles
    python scripts/evaluate_retrieval.py --profile medium --no-instruction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from contextforge.common.config import get_db_path
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.retrieval.retriever import DenseRetriever
from contextforge.storage.db import get_connection
from contextforge.storage.repositories import ChunkRepository


def load_qa_dataset(dataset_path: Path | str) -> list[dict]:
    """Load JSONL QA evaluation dataset."""
    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QA dataset not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            records.append(json.loads(line_str))
    return records


def evaluate_retrieval(
    profile: str,
    dataset_path: Path | str = "data/qa/retrieval_smoke.jsonl",
    index_base_dir: Path | str = "artifacts/faiss",
    db_path: Path | str | None = None,
    query_instruction_enabled: bool = True,
    embedder: BGEEmbedder | None = None,
) -> dict:
    """Evaluate retrieval metrics for a given chunk profile.

    Returns:
        dict with Recall@1, Recall@3, Recall@5, MRR, avg_latency_ms, and sample details.
    """
    qa_data = load_qa_dataset(dataset_path)
    if not qa_data:
        raise ValueError(f"Dataset '{dataset_path}' contains 0 questions.")

    index_dir = Path(index_base_dir) / profile
    target_db = Path(db_path) if db_path else get_db_path()

    store = FAISSStore(chunk_profile=profile)
    store.load(index_dir)

    if embedder is None:
        embedder = BGEEmbedder()

    conn = get_connection(target_db)
    chunk_repo = ChunkRepository(conn)

    retriever = DenseRetriever(
        embedder=embedder,
        store=store,
        chunk_repo=chunk_repo,
        top_k=5,
        query_instruction_enabled=query_instruction_enabled,
    )

    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    reciprocal_ranks = []
    latencies = []

    details = []

    for item in qa_data:
        qid = item["id"]
        question = item["question"]
        relevant_docs = set(item.get("relevant_document_ids", []))

        t0 = time.perf_counter()
        results = retriever.retrieve(question, top_k=5)
        latency = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(latency)

        retrieved_docs = [r.document_id for r in results]

        # Check hits
        hit_1 = bool(retrieved_docs[:1] and any(d in relevant_docs for d in retrieved_docs[:1]))
        hit_3 = bool(retrieved_docs[:3] and any(d in relevant_docs for d in retrieved_docs[:3]))
        hit_5 = bool(retrieved_docs[:5] and any(d in relevant_docs for d in retrieved_docs[:5]))

        if hit_1:
            r1_hits += 1
        if hit_3:
            r3_hits += 1
        if hit_5:
            r5_hits += 1

        # Calculate reciprocal rank
        rr = 0.0
        for rank_idx, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

        details.append(
            {
                "id": qid,
                "question": question,
                "relevant_docs": list(relevant_docs),
                "retrieved_docs": retrieved_docs,
                "hit_1": hit_1,
                "hit_3": hit_3,
                "hit_5": hit_5,
                "rr": rr,
                "latency_ms": latency,
            }
        )

    conn.close()

    total_q = len(qa_data)
    recall_1 = r1_hits / total_q
    recall_3 = r3_hits / total_q
    recall_5 = r5_hits / total_q
    mrr = sum(reciprocal_ranks) / total_q
    avg_latency = sum(latencies) / total_q

    return {
        "profile": profile,
        "query_instruction": query_instruction_enabled,
        "total_questions": total_q,
        "recall_1": recall_1,
        "recall_3": recall_3,
        "recall_5": recall_5,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "details": details,
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_retrieval",
        description="Evaluate ContextForge dense retrieval on smoke QA dataset.",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Profile to evaluate (default: medium).",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Evaluate all 3 profiles (small, medium, large).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qa/retrieval_smoke.jsonl",
        help="Path to evaluation questions (default: data/qa/retrieval_smoke.jsonl).",
    )
    parser.add_argument(
        "--no-instruction",
        action="store_true",
        help="Disable BGE query instruction prefix.",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    profiles = ["small", "medium", "large"] if args.all_profiles else [args.profile]

    print("\n========================================================")
    print("ContextForge Retrieval Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Instruction prefix: {'DISABLED' if args.no_instruction else 'ENABLED'}")
    print("========================================================\n")

    embedder = BGEEmbedder()
    results = []

    for p in profiles:
        print(f"Evaluating profile '{p}'...")
        res = evaluate_retrieval(
            profile=p,
            dataset_path=args.dataset,
            query_instruction_enabled=not args.no_instruction,
            embedder=embedder,
        )
        results.append(res)

    print("\n" + "=" * 70)
    print(f"{'Profile':<10} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} | {'Avg Latency':<12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['profile']:<10} | {r['recall_1']:<10.3f} | {r['recall_3']:<10.3f} | {r['recall_5']:<10.3f} | {r['mrr']:<8.3f} | {r['avg_latency_ms']:<8.1f} ms"
        )
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
