"""
Citation Extraction and Validation for ContextForge RAG Pipeline.
================================================================
Validates inline citation labels in generated answers against actual provided sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CitationValidationResult:
    """Outcome of validating citation labels in generated answer text."""

    used_labels: list[str] = field(default_factory=list)
    valid_labels: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no fabricated/unauthorized citation labels were detected."""
        return len(self.invalid_labels) == 0

    @property
    def has_citations(self) -> bool:
        """True if the answer referenced at least one citation label."""
        return len(self.used_labels) > 0


def extract_citation_labels(text: str) -> list[str]:
    """Extract citation labels (e.g. 'SOURCE_1', 'SOURCE_2') from text in order of appearance.

    Matches bracketed references like `[SOURCE_1]` or bare references like `SOURCE_1`.

    Args:
        text: Generated answer string.

    Returns:
        List of distinct citation label strings in order of first appearance.
    """
    if not text:
        return []

    # Find all occurrences of SOURCE_<digits> (with or without brackets)
    pattern = re.compile(r"\[?\b(SOURCE_\d+)\b\]?")
    matches = pattern.findall(text)

    # Preserve order of first appearance while deduplicating
    seen: set[str] = set()
    labels: list[str] = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            labels.append(match)

    return labels


def validate_citations(
    answer: str,
    allowed_labels: Iterable[str],
) -> CitationValidationResult:
    """Validate citations found in generated text against an allowed set of source labels.

    Args:
        answer: Generated answer text.
        allowed_labels: Iterable of valid source labels provided in context (e.g. {'SOURCE_1', 'SOURCE_2'}).

    Returns:
        CitationValidationResult detailing used, valid, and invalid citations.
    """
    allowed_set = set(allowed_labels)
    used_labels = extract_citation_labels(answer)

    valid_labels: list[str] = []
    invalid_labels: list[str] = []

    for label in used_labels:
        if label in allowed_set:
            valid_labels.append(label)
        else:
            invalid_labels.append(label)

    return CitationValidationResult(
        used_labels=used_labels,
        valid_labels=valid_labels,
        invalid_labels=invalid_labels,
    )
