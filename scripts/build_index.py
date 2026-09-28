"""
ContextForge Index Build CLI.
=============================
Extracts chunks from ingested documents, generates BGE embeddings, persists
chunk metadata in SQLite, and builds FAISS vector indices for semantic retrieval.

Usage:
    python scripts/build_index.py --profile medium
    python scripts/build_index.py --all
    python scripts/build_index.py --profile small --rebuild
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from contextforge.common.config import get_db_path
from contextforge.ingestion.chunker import TokenAwareChunker
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.storage.db import init_db
from contextforge.storage.repositories import ChunkRepository, DocumentRepository


def build_profile_index(
    profile: str,
    db_path: Path | str,
    output_base_dir: Path | str = "artifacts/faiss",
    rebuild: bool = True,
    chunker: TokenAwareChunker | None = None,
    embedder: BGEEmbedder | None = None,
) -> dict:
    """Build chunks and FAISS index for a specific profile.

    Args:
        profile: Profile name ("small", "medium", "large").
        db_path: Path to ContextForge SQLite database.
        output_base_dir: Base directory to save FAISS artifacts.
        rebuild: Whether to delete existing chunks in SQLite for this profile before saving.
        chunker: Shared TokenAwareChunker instance (optional).
        embedder: Shared BGEEmbedder instance (optional).

    Returns:
        Summary statistics dictionary.
    """
    t_start = time.perf_counter()
    conn = init_db(db_path)
    doc_repo = DocumentRepository(conn)
    chunk_repo = ChunkRepository(conn)

    docs = doc_repo.list_all()
    if not docs:
        conn.close()
        raise RuntimeError(f"No documents found in database '{db_path}'. Ingest corpus first.")

    if chunker is None:
        chunker = TokenAwareChunker()
    if embedder is None:
        embedder = BGEEmbedder()

    print(f"\n========================================================")
    print(f"Building index for profile: '{profile}'")
    print(f"Total documents to chunk: {len(docs)}")
    print(f"========================================================")

    # 1. Chunk all documents
    all_chunks = []
    doc_chunk_counts = {}
    for doc in docs:
        doc_chunks = chunker.chunk_document(doc, profile=profile)
        all_chunks.extend(doc_chunks)
        doc_chunk_counts[doc.id] = len(doc_chunks)

    total_chunks = len(all_chunks)
    if total_chunks == 0:
        conn.close()
        raise RuntimeError(f"Chunking produced 0 chunks for profile '{profile}'.")

    token_counts = [c.token_count for c in all_chunks]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / total_chunks

    print(f"Generated {total_chunks} chunks.")
    print(f"Token counts: min={min_tokens}, avg={avg_tokens:.1f}, max={max_tokens}")

    # 2. Persist chunks in SQLite
    if rebuild:
        deleted = chunk_repo.delete_by_profile(profile)
        if deleted > 0:
            print(f"Cleared {deleted} previous chunks for profile '{profile}'.")

    chunk_repo.save_chunks(all_chunks, embedding_model=embedder.model_name)
    print(f"Saved {total_chunks} chunks to SQLite 'chunks' table.")

    # 3. Compute dense embeddings
    print(f"Computing BGE embeddings for {total_chunks} passages (batch_size=32)...")
    texts = [c.text for c in all_chunks]
    t_embed_start = time.perf_counter()
    embeddings = embedder.embed_passages(texts, batch_size=32, show_progress_bar=True)
    embed_duration = time.perf_counter() - t_embed_start
    print(f"Embeddings computed in {embed_duration:.2f}s (shape={embeddings.shape}, dtype={embeddings.dtype}).")

    # 4. Build FAISS index
    chunk_ids = [c.id for c in all_chunks]
    store = FAISSStore(dimension=embedder.dimension, model_name=embedder.model_name, chunk_profile=profile)
    store.build(embeddings=embeddings, chunk_ids=chunk_ids, chunk_profile=profile)

    # 5. Persist FAISS artifacts
    profile_dir = Path(output_base_dir) / profile
    saved_path = store.save(profile_dir)
    total_duration = time.perf_counter() - t_start

    conn.close()

    summary = {
        "profile": profile,
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "min_tokens": min_tokens,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
        "embedding_shape": embeddings.shape,
        "embed_time_seconds": embed_duration,
        "total_time_seconds": total_duration,
        "artifact_dir": str(saved_path),
    }

    print(f"\nProfile '{profile}' completed successfully:")
    print(f"  Artifact directory: {saved_path}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Total vectors:      {store.total_vectors}")
    print(f"  Total wall time:    {total_duration:.2f}s")
    print(f"========================================================\n")

    return summary


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Build ContextForge dense FAISS indices from SQLite corpus.",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large"],
        default="medium",
        help="Chunking profile to build (default: medium).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all three chunking profiles (small, medium, large) sequentially.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: project data/contextforge.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/faiss",
        help="Base directory for FAISS artifacts (default: artifacts/faiss).",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Do not delete existing chunks in SQLite for the profile before saving.",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path) if args.db_path else get_db_path()

    if not db_path.is_file():
        print(f"Error: Database file '{db_path}' not found. Run ingestion first.", file=sys.stderr)
        return 1

    try:
        # Pre-initialize shared chunker and embedder for efficiency
        print("Loading tokenizer and BGE embedding model...")
        chunker = TokenAwareChunker()
        embedder = BGEEmbedder()

        profiles = ["small", "medium", "large"] if args.all else [args.profile]
        for p in profiles:
            build_profile_index(
                profile=p,
                db_path=db_path,
                output_base_dir=args.output_dir,
                rebuild=not args.no_rebuild,
                chunker=chunker,
                embedder=embedder,
            )

        print("Index build finished successfully.")
        return 0
    except Exception as e:
        print(f"\nIndex build error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
