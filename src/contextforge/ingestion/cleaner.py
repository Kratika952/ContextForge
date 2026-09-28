"""
Conservative text cleaning for ContextForge document ingestion.
================================================================
Performs normalization while strictly preserving punctuation, capitalization,
and technical terminology. Aggressive cleaning (lowercasing, punctuation stripping)
is forbidden because it destroys semantic meaning for cybersecurity QA.
"""

from __future__ import annotations

import re
import unicodedata

# Matches non-printable control characters excluding tab (0x09) and newline (0x0A)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Matches horizontal whitespace (spaces and tabs) of length 2 or more
_MULTI_SPACE_RE = re.compile(r"[^\S\r\n]+")

# Matches soft hyphen character
_SOFT_HYPHEN = "\xad"

# Matches a word split across a line break with a hyphen, where both sides are lowercase English letters
# e.g., "cyber-\nsecurity" -> "cybersecurity"
# Does NOT match capitalized terms, CVEs, or hyphenated compounds like "risk-based"
_LINE_SPLIT_WORD_RE = re.compile(r"(\b[a-z]{3,})-\n([a-z]{3,}\b)")

# Matches 3 or more consecutive newlines
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str, is_markdown: bool = False) -> str:
    """Conservatively clean extracted document text.

    Preserves:
    - All casing and punctuation (e.g. CVE-2025-1234, TLS 1.3, XSS, SQL queries, URLs).
    - Technical syntax, operators, brackets (< > / : _ -).
    - Code blocks and markdown formatting.

    Performs:
    - Unicode NFKC normalization (replaces ligatures like fi, fl with standard chars).
    - Normalizes line breaks (CRLF -> LF).
    - Removes non-printable control characters.
    - Replaces soft hyphens.
    - Fixes line-split English words (for PDF extracted text).
    - Collapses multiple horizontal spaces to single space without destroying newlines.
    - Limits consecutive blank lines to maximum 2 (preserving paragraph breaks).

    Args:
        text: Raw extracted text string.
        is_markdown: If True, skips PDF-specific line-wrap hyphen rejoining.

    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKC normalizes ligatures like fi, fl)
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Normalize line endings to standard LF
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip non-printable/control characters (preserving \n and \t)
    normalized = _CONTROL_CHAR_RE.sub("", normalized)

    # 4. Remove soft hyphens
    normalized = normalized.replace(_SOFT_HYPHEN, "")

    # 5. Fix line-break hyphenation for continuous prose (skip for markdown)
    if not is_markdown:
        normalized = _LINE_SPLIT_WORD_RE.sub(r"\1\2", normalized)

    # 6. Clean each line: collapse multiple horizontal spaces while trimming line edges
    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        line_clean = _MULTI_SPACE_RE.sub(" ", line).strip()
        cleaned_lines.append(line_clean)

    reconstructed = "\n".join(cleaned_lines)

    # 7. Collapse excessive vertical blank lines (keep max 2 newlines = paragraph break)
    reconstructed = _EXCESSIVE_NEWLINES_RE.sub("\n\n", reconstructed)

    # 8. Strip leading and trailing whitespace of the whole block
    return reconstructed.strip()
