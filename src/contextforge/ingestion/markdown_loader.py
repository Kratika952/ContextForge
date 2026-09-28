"""
Markdown extraction module for ContextForge.
============================================
Loads Markdown files (.md) preserving structural headings, technical formatting,
and UTF-8 content. Provides section-level provenance for RAG pipelines.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from contextforge.common.types import (
    DocumentExtractionError,
    DocumentNotFoundError,
    ExtractedDocument,
    ExtractedUnit,
    NoExtractableTextError,
)
from contextforge.ingestion.cleaner import clean_text

# Regex to detect top-level or sub-level markdown headings (# or ##)
_HEADING_SPLIT_RE = re.compile(r"(?m)^(#{1,3}\s+.+)$")


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class MarkdownLoader:
    """Loads and normalizes Markdown documentation files."""

    def __init__(self, min_char_threshold: int = 10) -> None:
        """Initialize MarkdownLoader.

        Args:
            min_char_threshold: Minimum characters required to consider the document valid.
        """
        self.min_char_threshold = min_char_threshold

    def load(self, file_path: Path | str) -> ExtractedDocument:
        """Load and extract text from a Markdown file.

        Args:
            file_path: Path to the .md file.

        Returns:
            ExtractedDocument containing section-level units, metadata, and checksum.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            DocumentExtractionError: If the file cannot be decoded as UTF-8.
            NoExtractableTextError: If the file contains no text.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise DocumentNotFoundError(f"Markdown file not found: {path}")

        checksum = compute_sha256(path)

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            try:
                # Fallback with latin-1 replacement if strict utf-8 fails
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as inner_exc:
                raise DocumentExtractionError(
                    f"Failed to read markdown file '{path.name}': {inner_exc}"
                ) from exc
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to read markdown file '{path.name}': {exc}"
            ) from exc

        cleaned_full = clean_text(raw_text, is_markdown=True)
        if len(cleaned_full) < self.min_char_threshold:
            raise NoExtractableTextError(
                f"Markdown file '{path.name}' contains only {len(cleaned_full)} character(s). "
                "Document has no extractable content."
            )

        # Extract title from the first level-1 heading or file stem
        title: str = path.stem.replace("_", " ").replace("-", " ").title()
        title_match = re.search(r"(?m)^#\s+(.+)$", raw_text)
        if title_match:
            title = title_match.group(1).strip()

        # Split into section-level units based on markdown headings
        # If heading pattern is found, split by headings; otherwise treat as single unit
        tokens = _HEADING_SPLIT_RE.split(raw_text)
        units: list[ExtractedUnit] = []

        if len(tokens) > 1:
            current_heading: str | None = None
            accumulated_body: list[str] = []
            unit_idx = 1

            for token in tokens:
                token_strip = token.strip()
                if not token_strip:
                    continue

                if _HEADING_SPLIT_RE.match(token_strip):
                    # Save previous section if it has content
                    if accumulated_body or current_heading:
                        section_text = clean_text("\n".join(accumulated_body), is_markdown=True)
                        if section_text or current_heading:
                            full_unit_text = f"{current_heading}\n\n{section_text}".strip() if current_heading else section_text
                            units.append(
                                ExtractedUnit(
                                    unit_index=unit_idx,
                                    heading=current_heading,
                                    text=full_unit_text,
                                    char_count=len(full_unit_text),
                                )
                            )
                            unit_idx += 1
                        accumulated_body = []
                    current_heading = token_strip
                else:
                    accumulated_body.append(token_strip)

            # Flush the last section
            if accumulated_body or current_heading:
                section_text = clean_text("\n".join(accumulated_body), is_markdown=True)
                full_unit_text = f"{current_heading}\n\n{section_text}".strip() if current_heading else section_text
                if full_unit_text:
                    units.append(
                        ExtractedUnit(
                            unit_index=unit_idx,
                            heading=current_heading,
                            text=full_unit_text,
                            char_count=len(full_unit_text),
                        )
                    )
        else:
            # Single unit document
            units.append(
                ExtractedUnit(
                    unit_index=1,
                    heading=title,
                    text=cleaned_full,
                    char_count=len(cleaned_full),
                )
            )

        if not units:
            units.append(
                ExtractedUnit(
                    unit_index=1,
                    heading=title,
                    text=cleaned_full,
                    char_count=len(cleaned_full),
                )
            )

        return ExtractedDocument(
            source_path=str(path),
            checksum=checksum,
            source_type="markdown",
            units=units,
            total_units=len(units),
            title=title,
            warnings=[],
            metadata={
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
            },
        )
