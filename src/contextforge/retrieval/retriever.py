"""
Dense Retriever for ContextForge.
=================================
Coordinates BGEEmbedder, FAISSStore, and SQLite ChunkRepository to execute
semantic similarity search and return standardized RetrievedChunk objects.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from contextforge.common.types import (
    Chunk,
    RetrievedChunk,
)
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.storage.db import get_connection
from contextforge.storage.repositories import ChunkRepository

DEFAULT_TOP_K = 5


def format_source_label(chunk: Chunk) -> str:
    """Format human-readable citation label for a retrieved chunk.

    Examples:
        PDF: [nist-csf-20-P12] or [nist-csf-20-P12-P13]
        Markdown: [owasp-a01-Broken-Access-Control]

    Args:
        chunk: Chunk domain model instance.

    Returns:
        Formatted source label string.
    """
    doc_id = chunk.document_id

    # PDF page-based citation
    if chunk.page_start is not None:
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            return f"[{doc_id}-P{chunk.page_start}-P{chunk.page_end}]"
        return f"[{doc_id}-P{chunk.page_start}]"

    # Markdown heading-based citation
    if chunk.section_heading:
        clean_heading = re.sub(r"^[#\s]+", "", chunk.section_heading).strip()
        slug = re.sub(r"[^\w\s-]", "", clean_heading)
        slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
        truncated_slug = slug[:30] if len(slug) > 30 else slug
        return f"[{doc_id}-{truncated_slug}]"

    return f"[{doc_id}-CHK{chunk.chunk_index:03d}]"


class DenseRetriever:
    """Dense retriever combining BGE embeddings, FAISS vector search, and SQLite metadata."""

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        store: FAISSStore | None = None,
        chunk_repo: ChunkRepository | None = None,
        db_path: Path | str | None = None,
        conn: sqlite3.Connection | None = None,
        top_k: int = DEFAULT_TOP_K,
        query_instruction_enabled: bool = True,
    ) -> None:
        """Initialize DenseRetriever.

        Args:
            embedder: BGEEmbedder instance. If None, default instance is created.
            store: FAISSStore instance. If None, default instance is created.
            chunk_repo: ChunkRepository instance.
            db_path: Database path if chunk_repo is None.
            conn: Existing connection if chunk_repo is None.
            top_k: Default number of passages to retrieve.
            query_instruction_enabled: Whether to apply BGE retrieval instruction to queries.
        """
        self.embedder = embedder or BGEEmbedder()
        self.store = store or FAISSStore()
        self.top_k = top_k
        self.query_instruction_enabled = query_instruction_enabled

        if chunk_repo is not None:
            self.chunk_repo = chunk_repo
            self._conn = chunk_repo.conn
            self._owns_conn = False
        elif conn is not None:
            self._conn = conn
            self.chunk_repo = ChunkRepository(conn)
            self._owns_conn = False
        else:
            self._conn = get_connection(db_path)
            self.chunk_repo = ChunkRepository(self._conn)
            self._owns_conn = True

    def close(self) -> None:
        """Close SQLite database connection if owned by retriever."""
        if self._owns_conn and self._conn:
            self._conn.close()

    def __enter__(self) -> DenseRetriever:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Execute semantic search for a query and return ranked RetrievedChunk objects.

        Args:
            query: Plain-text search query.
            top_k: Number of chunks to retrieve (defaults to self.top_k).

        Returns:
            List of RetrievedChunk instances ranked by similarity score.
        """
        k = top_k if top_k is not None else self.top_k
        query_text = query.strip()
        if not query_text or not self.store.is_built:
            return []

        # 1. Embed query
        query_vector = self.embedder.embed_query(
            query_text,
            add_instruction=self.query_instruction_enabled,
        )

        # 2. Search FAISS index
        search_results = self.store.search(query_vector, top_k=k)
        if not search_results:
            return []

        # 3. Retrieve chunks from SQLite repository
        chunk_ids = [cid for cid, _, _ in search_results]
        chunks = self.chunk_repo.get_by_ids(chunk_ids)
        chunk_map = {c.id: c for c in chunks}

        # 4. Construct ranked RetrievedChunk instances
        retrieved: list[RetrievedChunk] = []
        for chunk_id, score, rank in search_results:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            source_label = format_source_label(chunk)
            metadata = {
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_heading": chunk.section_heading,
                "chunk_profile": chunk.chunk_profile,
                **chunk.metadata,
            }

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    source_label=source_label,
                    metadata=metadata,
                )
            )

        return retrieved
