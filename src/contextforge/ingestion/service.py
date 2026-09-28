"""
Ingestion Service for ContextForge.
==================================
Coordinates document validation, loader dispatch, duplicate prevention via SHA-256,
and SQLite metadata persistence without ML dependencies.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextforge.common.types import (
    Document,
    DocumentNotFoundError,
    DuplicateDocumentError,
    ExtractedDocument,
    UnsupportedDocumentTypeError,
)
from contextforge.ingestion.markdown_loader import MarkdownLoader
from contextforge.ingestion.pdf_loader import PDFLoader, compute_sha256
from contextforge.storage.db import get_connection, init_db
from contextforge.storage.repositories import DocumentRepository


@dataclass
class IngestionResult:
    """Outcome of an ingestion attempt."""

    status: str  # "INGESTED" | "ALREADY_INGESTED"
    document: Document
    extracted: ExtractedDocument | None = None
    message: str = ""

    @property
    def is_new(self) -> bool:
        """True if newly inserted, False if duplicate skipped."""
        return self.status == "INGESTED"


def slugify(text: str) -> str:
    """Generate a clean URL/ID friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


class IngestionService:
    """Coordinates document ingestion into the ContextForge repository."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        conn: sqlite3.Connection | None = None,
        repo: DocumentRepository | None = None,
    ) -> None:
        """Initialize IngestionService.

        Args:
            db_path: Path to SQLite DB. If None, uses project default.
            conn: Existing sqlite3.Connection (takes precedence over db_path).
            repo: Existing DocumentRepository instance (takes precedence).
        """
        if repo is not None:
            self.repo = repo
            self._conn = repo.conn
            self._owns_conn = False
        elif conn is not None:
            self._conn = conn
            self.repo = DocumentRepository(conn)
            self._owns_conn = False
        else:
            self._conn = init_db(db_path)
            self.repo = DocumentRepository(self._conn)
            self._owns_conn = True

        self.pdf_loader = PDFLoader()
        self.markdown_loader = MarkdownLoader()

    def close(self) -> None:
        """Close database connection if owned by service."""
        if self._owns_conn and self._conn:
            self._conn.close()

    def __enter__(self) -> IngestionService:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def ingest_file(
        self,
        file_path: Path | str,
        document_id: str | None = None,
        title: str | None = None,
        on_duplicate: str = "skip",  # "skip" | "raise"
    ) -> IngestionResult:
        """Ingest a single document file (PDF or Markdown).

        Flow:
        path -> validate -> determine type -> SHA-256 -> check duplicate ->
        load -> extract -> clean -> persist -> IngestionResult

        Args:
            file_path: Path to target file.
            document_id: Optional custom identifier. If None, generated from title/path.
            title: Optional custom title override.
            on_duplicate: Behavior when file checksum already exists:
                "skip": return IngestionResult(status="ALREADY_INGESTED")
                "raise": raise DuplicateDocumentError

        Returns:
            IngestionResult with status and document details.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            UnsupportedDocumentTypeError: If the file extension is not .pdf or .md.
            DuplicateDocumentError: If duplicate exists and on_duplicate="raise".
            DocumentExtractionError: If parsing fails.
            NoExtractableTextError: If document contains no extractable text.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise DocumentNotFoundError(f"File not found: {path}")

        # Determine supported source type
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            source_type = "pdf"
            loader = self.pdf_loader
        elif suffix in (".md", ".markdown"):
            source_type = "markdown"
            loader = self.markdown_loader
        else:
            raise UnsupportedDocumentTypeError(
                f"Unsupported file type '{suffix}' for file '{path.name}'. "
                "Supported formats are PDF (.pdf) and Markdown (.md, .markdown)."
            )

        # Compute SHA-256 artifact checksum
        checksum = compute_sha256(path)

        # Check for duplicate
        existing = self.repo.get_by_checksum(checksum)
        if existing is not None:
            if on_duplicate == "raise":
                raise DuplicateDocumentError(
                    f"Document with checksum '{checksum}' has already been ingested "
                    f"(id: '{existing.id}', title: '{existing.title}')."
                )
            return IngestionResult(
                status="ALREADY_INGESTED",
                document=existing,
                message=f"Duplicate document detected (id: '{existing.id}'). Skipped database insert.",
            )

        # Extract and clean document content
        extracted = loader.load(path)

        # Determine title and ID
        final_title = title or extracted.title or path.stem
        final_id = document_id or slugify(path.stem)

        # Build Document entity
        doc = Document(
            id=final_id,
            title=final_title,
            source_path=str(path),
            checksum=checksum,
            source_type=source_type,
            status="INGESTED",
            metadata={
                "filename": path.name,
                "total_units": extracted.total_units,
                "total_chars": extracted.total_char_count,
                "warnings": extracted.warnings,
                "loader_metadata": extracted.metadata,
            },
        )

        # Persist to SQLite
        self.repo.save(doc)

        return IngestionResult(
            status="INGESTED",
            document=doc,
            extracted=extracted,
            message=f"Successfully ingested {source_type.upper()} document '{final_title}' "
            f"({extracted.total_units} units, {extracted.total_char_count} chars).",
        )

    def ingest_directory(
        self,
        dir_path: Path | str,
        recursive: bool = True,
        on_duplicate: str = "skip",
    ) -> list[IngestionResult]:
        """Ingest all supported documents in a directory.

        Args:
            dir_path: Directory path containing documents.
            recursive: If True, searches subdirectories.
            on_duplicate: "skip" or "raise".

        Returns:
            List of IngestionResults for all found files.
        """
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise DocumentNotFoundError(f"Directory not found: {path}")

        pattern = "**/*" if recursive else "*"
        candidates = sorted(
            [
                p
                for p in path.glob(pattern)
                if p.is_file() and p.suffix.lower() in (".pdf", ".md", ".markdown")
            ]
        )

        results: list[IngestionResult] = []
        for file_p in candidates:
            # Skip manifest files or gitkeeps if matched
            if file_p.name.startswith(".") or file_p.name == "manifest.json":
                continue
            res = self.ingest_file(file_p, on_duplicate=on_duplicate)
            results.append(res)

        return results
