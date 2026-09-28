"""
Integration tests for ContextForge Retrieval Layer.
==================================================
Tests end-to-end retrieval against real SQLite metadata and persisted FAISS indices.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from contextforge.common.types import RetrievedChunk
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.retrieval.retriever import DenseRetriever
from contextforge.storage.db import get_connection
from contextforge.storage.repositories import ChunkRepository


@pytest.mark.integration
def test_retrieval_integration_medium_profile():
    index_dir = Path("artifacts/faiss/medium")
    db_path = Path("data/contextforge.db")

    if not index_dir.is_dir() or not db_path.is_file():
        pytest.skip("FAISS index or database not found. Run scripts/build_index.py first.")

    store = FAISSStore(chunk_profile="medium")
    store.load(index_dir)
    assert store.total_vectors > 0

    embedder = BGEEmbedder()
    conn = get_connection(db_path)
    chunk_repo = ChunkRepository(conn)

    retriever = DenseRetriever(
        embedder=embedder,
        store=store,
        chunk_repo=chunk_repo,
        top_k=3,
        query_instruction_enabled=True,
    )

    query = "How to prevent Broken Access Control vulnerabilities?"
    results = retriever.retrieve(query, top_k=3)

    conn.close()

    assert len(results) == 3
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.chunk_id
        assert r.document_id
        assert r.text
        assert r.score > 0.5
        assert r.source_label.startswith("[") and r.source_label.endswith("]")

    # Verify rank order
    assert results[0].score >= results[1].score >= results[2].score
    assert [r.rank for r in results] == [1, 2, 3]


@pytest.mark.integration
def test_search_cli_execution():
    index_dir = Path("artifacts/faiss/medium")
    if not index_dir.is_dir():
        pytest.skip("Index not built.")

    cmd = [
        sys.executable,
        "scripts/search.py",
        "What is the NIST CSF Govern function?",
        "--profile",
        "medium",
        "-k",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "Query: 'What is the NIST CSF Govern function?'" in res.stdout
    assert "Results retrieved: 2" in res.stdout
    assert "Rank 1" in res.stdout
