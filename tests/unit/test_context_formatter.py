"""
Unit Tests for ContextFormatter.
================================
Tests deterministic label generation, provenance extraction, and token budgeting.
"""

from __future__ import annotations

import pytest

from contextforge.common.types import RetrievedChunk
from contextforge.llm.context_formatter import (
    ContextFormatter,
    resolve_document_source_name,
    resolve_reference_label,
)


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    score: float = 0.85,
    rank: int = 1,
    page_start: int | None = None,
    page_end: int | None = None,
    heading: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        score=score,
        rank=rank,
        source_label=f"[{doc_id}]",
        metadata={
            "page_start": page_start,
            "page_end": page_end,
            "section_heading": heading,
            "token_count": len(text.split()),
        },
    )


def test_resolve_document_source_name():
    assert resolve_document_source_name("nist-csf-20") == "NIST CSF 2.0"
    assert resolve_document_source_name("cisa-cpg-20") == "CISA CPG 2.0"
    assert resolve_document_source_name("owasp-a01-2025") == "OWASP Top 10:2025"
    assert resolve_document_source_name("custom_doc", {"title": "Custom Title"}) == "Custom Title"


def test_resolve_reference_label():
    c1 = _make_chunk("c1", "nist-csf-20", "Text", page_start=8, page_end=8)
    assert resolve_reference_label(c1) == "Page 8"

    c2 = _make_chunk("c2", "nist-csf-20", "Text", page_start=8, page_end=9)
    assert resolve_reference_label(c2) == "Pages 8–9"

    c3 = _make_chunk("c3", "owasp-a01", "Text", heading="## How to Prevent")
    assert resolve_reference_label(c3) == "How to Prevent"


def test_formatter_empty_chunks():
    formatter = ContextFormatter(max_context_tokens=500)
    result = formatter.format([])
    assert result.context_text == ""
    assert result.included_chunks == []
    assert result.context_token_count == 0
    assert result.retrieved_count == 0
    assert result.included_count == 0


def test_formatter_deterministic_labels_and_provenance():
    formatter = ContextFormatter(max_context_tokens=1000)
    c1 = _make_chunk("c1", "owasp-a01", "Prevent access control issues.", score=0.9, rank=1, heading="Prevention")
    c2 = _make_chunk("c2", "nist-csf-20", "Govern function establishes policy.", score=0.8, rank=2, page_start=12)

    result = formatter.format([c1, c2])

    assert result.retrieved_count == 2
    assert result.included_count == 2
    assert "SOURCE_1" in result.label_to_chunk
    assert "SOURCE_2" in result.label_to_chunk
    assert result.label_to_chunk["SOURCE_1"].chunk_id == "c1"
    assert result.label_to_chunk["SOURCE_2"].chunk_id == "c2"

    assert "[SOURCE_1]" in result.context_text
    assert "Source: OWASP Top 10:2025" in result.context_text
    assert "Reference: Prevention" in result.context_text

    assert "[SOURCE_2]" in result.context_text
    assert "Source: NIST CSF 2.0" in result.context_text
    assert "Reference: Page 12" in result.context_text


def test_formatter_enforces_token_budget_without_mid_passage_truncation():
    # Mock token counter: each word is 10 tokens
    formatter = ContextFormatter(
        max_context_tokens=50,
        token_counter=lambda text: len(text.split()) * 10,
    )

    c1 = _make_chunk("c1", "doc1", "short passage one", score=0.9, rank=1)  # ~6 words = ~60 tokens with header?
    # Let's use simple token counting
    def mock_counter(t: str) -> int:
        return len(t.split())

    formatter = ContextFormatter(
        max_context_tokens=15,  # Each passage block with header has ~8 words
        token_counter=mock_counter,
    )

    c1 = _make_chunk("c1", "doc1", "word1 word2", score=0.9, rank=1)
    c2 = _make_chunk("c2", "doc2", "word3 word4 word5 word6 word7 word8 word9 word10", score=0.8, rank=2)

    result = formatter.format([c1, c2])

    # First chunk fits, second chunk would exceed budget (15 words) so it's skipped
    assert result.retrieved_count == 2
    assert result.included_count == 1
    assert "SOURCE_1" in result.label_to_chunk
    assert "SOURCE_2" not in result.label_to_chunk
    assert "[SOURCE_1]" in result.context_text
    assert "[SOURCE_2]" not in result.context_text
    assert result.context_token_count <= 15
