"""
Unit Tests for Citation Extraction and Validation.
==================================================
Tests valid citations, invalid citations, and un-cited answers.
"""

from __future__ import annotations

import pytest

from contextforge.llm.citations import extract_citation_labels, validate_citations


def test_extract_citation_labels_bracketed():
    text = "According to [SOURCE_1], access control is vital. Furthermore, [SOURCE_2] discusses MFA."
    labels = extract_citation_labels(text)
    assert labels == ["SOURCE_1", "SOURCE_2"]


def test_extract_citation_labels_unbracketed():
    text = "SOURCE_1 provides guidance on password policies while SOURCE_3 discusses logging."
    labels = extract_citation_labels(text)
    assert labels == ["SOURCE_1", "SOURCE_3"]


def test_extract_citation_labels_deduplicated_order_preserved():
    text = "See [SOURCE_2], then [SOURCE_1], and again [SOURCE_2]."
    labels = extract_citation_labels(text)
    assert labels == ["SOURCE_2", "SOURCE_1"]


def test_validate_citations_all_valid():
    text = "Broken access control is prevented by role enforcement [SOURCE_1] and least privilege [SOURCE_2]."
    allowed = ["SOURCE_1", "SOURCE_2", "SOURCE_3"]
    result = validate_citations(text, allowed)

    assert result.is_valid is True
    assert result.has_citations is True
    assert result.used_labels == ["SOURCE_1", "SOURCE_2"]
    assert result.valid_labels == ["SOURCE_1", "SOURCE_2"]
    assert result.invalid_labels == []


def test_validate_citations_detects_invalid():
    text = "According to [SOURCE_1] and [SOURCE_9], SQL injection requires parameterized queries."
    allowed = ["SOURCE_1", "SOURCE_2"]
    result = validate_citations(text, allowed)

    assert result.is_valid is False
    assert result.has_citations is True
    assert result.used_labels == ["SOURCE_1", "SOURCE_9"]
    assert result.valid_labels == ["SOURCE_1"]
    assert result.invalid_labels == ["SOURCE_9"]


def test_validate_citations_no_citations():
    text = "The supplied cybersecurity sources do not provide enough information to answer this question."
    allowed = ["SOURCE_1", "SOURCE_2"]
    result = validate_citations(text, allowed)

    assert result.is_valid is True
    assert result.has_citations is False
    assert result.used_labels == []
    assert result.valid_labels == []
    assert result.invalid_labels == []
