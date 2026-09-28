"""
Unit tests for ContextForge TokenAwareChunker.
==============================================
Verifies token-bounded document segmentation, provenance tracking,
profile configurations, and deterministic chunk IDs.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from contextforge.common.types import (
    Chunk,
    Document,
    ExtractedDocument,
    ExtractedUnit,
)
from contextforge.ingestion.chunker import DEFAULT_PROFILES, TokenAwareChunker


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer with word-based token simulation."""
    tok = MagicMock()
    tok.model_max_length = 512

    def fake_encode(text: str, add_special_tokens: bool = False):
        if not text:
            return []
        return [hash(w) % 10000 for w in text.split()]

    def fake_decode(tokens, skip_special_tokens: bool = True):
        return f"decoded_{len(tokens)}_tokens"

    tok.encode.side_effect = fake_encode
    tok.decode.side_effect = fake_decode
    return tok


def test_chunker_initialization_with_profiles(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)
    assert "small" in chunker.profiles
    assert "medium" in chunker.profiles
    assert "large" in chunker.profiles
    assert chunker.profiles["medium"]["chunk_size_tokens"] == 300


def test_chunk_text_convenience_method(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)
    sample_text = "Word " * 50
    chunks = chunker.chunk_text(sample_text, document_id="doc1", profile="small")

    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc1"
    assert chunks[0].chunk_profile == "small"
    assert chunks[0].token_count > 0
    assert chunks[0].id.startswith("doc1_small_")


def test_chunk_document_preserves_provenance(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)

    doc = Document(
        id="nist-test",
        title="NIST Test Document",
        source_path=":test:",
        checksum="abcd1234",
        source_type="pdf",
    )
    extracted = ExtractedDocument(
        source_path=":test:",
        checksum="abcd1234",
        source_type="pdf",
        units=[
            ExtractedUnit(unit_index=1, text="Page one text content for security testing.", char_count=45),
            ExtractedUnit(unit_index=2, text="Page two defensive controls and framework mapping.", char_count=52),
        ],
        total_units=2,
        title=doc.title,
    )

    chunks = chunker.chunk_document(doc, extracted=extracted, profile="medium")

    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.document_id == "nist-test"
        assert c.chunk_profile == "medium"
        assert c.content_hash
        assert c.token_count > 0
        assert c.metadata["document_title"] == "NIST Test Document"


def test_chunk_document_preserves_markdown_heading(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)

    doc = Document(
        id="owasp-test",
        title="OWASP Broken Access Control",
        source_path=":test.md:",
        checksum="beef5678",
        source_type="markdown",
    )
    extracted = ExtractedDocument(
        source_path=":test.md:",
        checksum="beef5678",
        source_type="markdown",
        units=[
            ExtractedUnit(
                unit_index=1,
                text="Deny by default is the primary defense against unauthorized access.",
                char_count=67,
                heading="## How to Prevent",
            ),
        ],
        total_units=1,
        title=doc.title,
    )

    chunks = chunker.chunk_document(doc, extracted=extracted, profile="small")

    assert len(chunks) == 1
    assert chunks[0].section_heading == "## How to Prevent"


def test_chunk_document_raises_on_invalid_profile(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)
    doc = Document(id="d", title="t", source_path="p", checksum="c")
    extracted = ExtractedDocument(source_path="p", checksum="c", source_type="pdf", units=[], total_units=0)

    with pytest.raises(ValueError, match="Unknown chunking profile"):
        chunker.chunk_document(doc, extracted=extracted, profile="ultra_tiny")


def test_no_empty_chunks(mock_tokenizer):
    chunker = TokenAwareChunker(tokenizer=mock_tokenizer)
    doc = Document(id="empty-doc", title="Empty", source_path="p", checksum="c")
    extracted = ExtractedDocument(
        source_path="p",
        checksum="c",
        source_type="pdf",
        units=[ExtractedUnit(unit_index=1, text="   \n\n   ", char_count=0)],
        total_units=1,
    )

    chunks = chunker.chunk_document(doc, extracted=extracted, profile="medium")
    assert chunks == []
