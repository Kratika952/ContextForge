"""
ContextForge Semantic Search CLI.
=================================
Performs dense retrieval against persisted FAISS vector index using
BGE-small-en-v1.5 and outputs ranked source chunks with scores and provenance.

Usage:
    python scripts/search.py "How is SQL injection prevented?"
    python scripts/search.py "What is the CSF 2.0 Govern function?" --profile small -k 3
    python scripts/search.py "What are cross-sector cybersecurity performance goals?" --no-instruction
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from contextforge.common.config import get_db_path
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.retrieval.retriever import DenseRetriever
from contextforge.storage.db import get_connection
from contextforge.storage.repositories import ChunkRepository


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="search",
        description="Search ContextForge knowledge base using dense vector retrieval.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search question or query.",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Chunking profile index to query (default: medium).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Custom index directory. Defaults to artifacts/faiss/<profile>.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: data/contextforge.db).",
    )
    parser.add_argument(
        "--no-instruction",
        action="store_true",
        help="Disable BGE query instruction prefix.",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path) if args.db_path else get_db_path()
    index_dir = Path(args.index_dir) if args.index_dir else Path("artifacts/faiss") / args.profile

    if not db_path.is_file():
        print(f"Error: Database not found at '{db_path}'.", file=sys.stderr)
        return 1

    if not index_dir.is_dir():
        print(
            f"Error: FAISS index not found at '{index_dir}'. "
            f"Run 'python scripts/build_index.py --profile {args.profile}' first.",
            file=sys.stderr,
        )
        return 1

    t0 = time.perf_counter()

    # Load FAISS index
    store = FAISSStore(chunk_profile=args.profile)
    store.load(index_dir)

    # Load embedder and SQLite repository
    embedder = BGEEmbedder()
    conn = get_connection(db_path)
    chunk_repo = ChunkRepository(conn)

    retriever = DenseRetriever(
        embedder=embedder,
        store=store,
        chunk_repo=chunk_repo,
        top_k=args.top_k,
        query_instruction_enabled=not args.no_instruction,
    )

    t_init = time.perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    t_search = time.perf_counter() - t_init

    conn.close()

    print(f"\n========================================================")
    print(f"Query: '{args.query}'")
    print(f"Index: {index_dir} (Profile: {args.profile}, Total vectors: {store.total_vectors})")
    print(f"Query instruction: {'DISABLED' if args.no_instruction else 'ENABLED'}")
    print(f"Results retrieved: {len(results)} in {t_search * 1000:.1f}ms")
    print(f"========================================================\n")

    if not results:
        print("No matching passages found.")
        return 0

    for res in results:
        print(f"--- [Rank {res.rank}] Score: {res.score:.4f} | Label: {res.source_label} ---")
        print(f"Document: {res.document_id} | Chunk ID: {res.chunk_id}")
        meta = res.metadata
        provenance = []
        if meta.get("page_start"):
            provenance.append(f"Pages: {meta['page_start']}-{meta.get('page_end', meta['page_start'])}")
        if meta.get("section_heading"):
            provenance.append(f"Section: {meta['section_heading']}")
        if meta.get("token_count"):
            provenance.append(f"Tokens: {meta['token_count']}")
        if provenance:
            print(f"Provenance: {' | '.join(provenance)}")

        # Preview snippet (first ~300 chars)
        preview = res.text.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:297] + "..."
        print(f"Content:\n  {preview}\n")

    print(f"Search complete in {(time.perf_counter() - t0):.2f}s total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
