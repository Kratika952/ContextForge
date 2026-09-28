"""
PDF extraction module using PyMuPDF (fitz).
============================================
Extracts text from PDF documents while preserving page-level provenance,
computing SHA-256 checksums, and detecting scanned/empty PDFs that require OCR.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf

from contextforge.common.types import (
    DocumentExtractionError,
    DocumentNotFoundError,
    ExtractedDocument,
    ExtractedUnit,
    NoExtractableTextError,
)
from contextforge.ingestion.cleaner import clean_text


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Hex-encoded lowercase SHA-256 string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class PDFLoader:
    """Extracts text and metadata from PDF files using PyMuPDF."""

    def __init__(self, min_char_threshold: int = 10) -> None:
        """Initialize PDFLoader.

        Args:
            min_char_threshold: Minimum total characters required across the document
                to consider it text-based rather than scanned/empty.
        """
        self.min_char_threshold = min_char_threshold

    def load(self, file_path: Path | str) -> ExtractedDocument:
        """Load and extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ExtractedDocument containing page-by-page text, metadata, and checksum.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            DocumentExtractionError: If the file cannot be opened or parsed as a valid PDF.
            NoExtractableTextError: If the PDF contains no extractable text (e.g. scanned image).
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise DocumentNotFoundError(f"PDF file not found: {path}")

        checksum = compute_sha256(path)

        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to open or parse PDF '{path.name}': {exc}"
            ) from exc

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise NoExtractableTextError(
                    f"PDF '{path.name}' contains 0 pages."
                )

            doc_metadata: dict[str, Any] = {}
            if doc.metadata:
                doc_metadata = {
                    k: v for k, v in doc.metadata.items() if v and isinstance(v, str)
                }

            raw_title = doc_metadata.get("title")
            title = raw_title.strip() if raw_title and raw_title.strip() else path.stem

            units: list[ExtractedUnit] = []
            warnings: list[str] = []
            empty_page_indices: list[int] = []

            for idx, page in enumerate(doc):
                page_num = idx + 1
                try:
                    raw_page_text = page.get_text("text")
                except Exception as exc:
                    warnings.append(f"Failed to extract text from page {page_num}: {exc}")
                    raw_page_text = ""

                cleaned = clean_text(raw_page_text, is_markdown=False)
                char_count = len(cleaned)

                if char_count == 0:
                    empty_page_indices.append(page_num)

                units.append(
                    ExtractedUnit(
                        unit_index=page_num,
                        text=cleaned,
                        char_count=char_count,
                    )
                )

            total_chars = sum(u.char_count for u in units)

            # Check if this document is scanned / image-only / empty
            if total_chars < self.min_char_threshold:
                raise NoExtractableTextError(
                    f"PDF '{path.name}' contains only {total_chars} extractable character(s) across "
                    f"{total_pages} page(s). This appears to be a scanned or image-only PDF. "
                    "OCR is required to process this document, which is unsupported in this pipeline."
                )

            if empty_page_indices and len(empty_page_indices) < total_pages:
                warnings.append(
                    f"{len(empty_page_indices)} page(s) contained no extractable text: {empty_page_indices}"
                )

            return ExtractedDocument(
                source_path=str(path),
                checksum=checksum,
                source_type="pdf",
                units=units,
                total_units=total_pages,
                title=title,
                warnings=warnings,
                metadata={
                    "filename": path.name,
                    "file_size_bytes": path.stat().st_size,
                    "pdf_metadata": doc_metadata,
                },
            )
        finally:
            doc.close()
