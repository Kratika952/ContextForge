"""
ContextForge common data types and custom exceptions.
=====================================================
Core data contracts shared across ingestion, storage, retrieval, and pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Exceptions ───────────────────────────────────────────────────────────────


class ContextForgeError(Exception):
    """Base exception for all ContextForge domain errors."""


class IngestionError(ContextForgeError):
    """Base exception for document ingestion failures."""


class DocumentNotFoundError(IngestionError, FileNotFoundError):
    """Raised when a target document file cannot be found."""


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when a document has an unsupported extension or media type."""


class DocumentExtractionError(IngestionError):
    """Raised when extraction fails due to file corruption or parsing errors."""


class NoExtractableTextError(IngestionError):
    """Raised when a document contains no extractable text (e.g. scanned/image-only PDF)."""


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical checksum has already been ingested."""


class RetrievalError(ContextForgeError):
    """Base exception for retrieval or indexing errors."""


class IndexNotFoundError(RetrievalError, FileNotFoundError):
    """Raised when a requested FAISS index artifact cannot be found."""


class IncompatibleIndexError(RetrievalError):
    """Raised when an index metadata check fails (e.g. model mismatch, dimension mismatch)."""


class EmbeddingError(RetrievalError):
    """Raised when text embedding fails."""


# ── Ingestion Data Contracts ──────────────────────────────────────────────────


@dataclass
class Document:
    """Document record matching Section 6.1 of IMPLEMENTATION_PLAN.md.

    Represents an authoritative document persisted in the metadata database.
    """

    id: str
    title: str
    source_path: str
    checksum: str
    source_type: str = "pdf"  # "pdf" | "markdown"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "INGESTED"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedUnit:
    """Single page or section extracted from a document with provenance tracking."""

    unit_index: int  # 1-indexed page number (PDF) or section index (Markdown)
    text: str
    char_count: int
    heading: str | None = None  # Section title or heading for Markdown

    @property
    def page_number(self) -> int:
        """Alias for unit_index when working with page-based documents (PDF)."""
        return self.unit_index


# Backward-compatible alias for PDF-specific workflows
ExtractedPage = ExtractedUnit


@dataclass
class ExtractedDocument:
    """Complete normalized extracted document before database persistence."""

    source_path: str
    checksum: str
    source_type: str
    units: list[ExtractedUnit]
    total_units: int
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pages(self) -> list[ExtractedUnit]:
        """Alias for units for backward compatibility with PDF tests."""
        return self.units

    @property
    def total_pages(self) -> int:
        """Alias for total_units for backward compatibility with PDF tests."""
        return self.total_units

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages/sections separated by double newlines."""
        return "\n\n".join(unit.text for unit in self.units if unit.text)

    @property
    def total_char_count(self) -> int:
        """Total characters extracted across all units."""
        return sum(unit.char_count for unit in self.units)


# ── Chunking & Retrieval Data Contracts ──────────────────────────────────────


@dataclass
class Chunk:
    """Segmented text chunk with provenance tracking matching Section 6.2 of IMPLEMENTATION_PLAN.md."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    content_hash: str
    chunk_profile: str = "medium"
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """Retrieved chunk matching Section 6.3 of IMPLEMENTATION_PLAN.md."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    source_label: str
    metadata: dict[str, Any] = field(default_factory=dict)
