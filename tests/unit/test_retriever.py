"""
Unit tests for ContextForge DenseRetriever.
===========================================
Verifies query execution, citation source labeling, and result mapping.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from contextforge.common.types import Chunk, RetrievedChunk
from contextforge.retrieval.retriever import DenseRetriever, format_source_label


def test_format_source_label_pdf():
    c1 = Chunk(
        id="c1",
        document_id="nist-csf-20",
        chunk_index=5,
        text="text",
        token_count=100,
        content_hash="h1",
        page_start=12,
        page_end=12,
    )
    assert format_source_label(c1) == "[nist-csf-20-P12]"

    c2 = Chunk(
        id="c2",
        document_id="cisa-cpg-20",
        chunk_index=8,
        text="text",
        token_count=120,
        content_hash="h2",
        page_start=15,
        page_end=16,
    )
    assert format_source_label(c2) == "[cisa-cpg-20-P15-P16]"


def test_format_source_label_markdown():
    c = Chunk(
        id="c3",
        document_id="a01-2025-broken-access-control",
        chunk_index=2,
        text="text",
        token_count=90,
        content_hash="h3",
        section_heading="## How to Prevent",
    )
    assert format_source_label(c) == "[a01-2025-broken-access-control-How-to-Prevent]"


def test_dense_retriever_retrieve_empty_query():
    retriever = DenseRetriever(
        embedder=MagicMock(),
        store=MagicMock(),
        chunk_repo=MagicMock(),
    )
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_dense_retriever_flow():
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = np.zeros((1, 384), dtype=np.float32)

    mock_store = MagicMock()
    mock_store.is_built = True
    mock_store.search.return_value = [("chk_01", 0.92, 1), ("chk_02", 0.85, 2)]

    chunk1 = Chunk(
        id="chk_01",
        document_id="doc1",
        chunk_index=0,
        text="Evidence one on defensive security.",
        token_count=50,
        content_hash="h1",
        page_start=3,
        page_end=3,
    )
    chunk2 = Chunk(
        id="chk_02",
        document_id="doc2",
        chunk_index=1,
        text="Evidence two on vulnerability mitigation.",
        token_count=60,
        content_hash="h2",
        section_heading="# Mitigation",
    )

    mock_repo = MagicMock()
    mock_repo.get_by_ids.return_value = [chunk1, chunk2]

    retriever = DenseRetriever(
        embedder=mock_embedder,
        store=mock_store,
        chunk_repo=mock_repo,
        top_k=2,
    )

    results = retriever.retrieve("What is defensive security?")

    assert len(results) == 2
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "chk_01"
    assert results[0].score == 0.92
    assert results[0].rank == 1
    assert results[0].source_label == "[doc1-P3]"

    assert results[1].chunk_id == "chk_02"
    assert results[1].score == 0.85
    assert results[1].rank == 2
    assert results[1].source_label == "[doc2-Mitigation]"
