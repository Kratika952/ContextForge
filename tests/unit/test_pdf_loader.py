"""Unit tests for PDF extraction with PyMuPDF."""

from pathlib import Path
import pymupdf
import pytest

from contextforge.common.types import (
    DocumentExtractionError,
    DocumentNotFoundError,
    NoExtractableTextError,
)
from contextforge.ingestion.pdf_loader import PDFLoader, compute_sha256


def create_test_pdf(path: Path, pages_text: list[str]) -> Path:
    """Helper to create a deterministic test PDF fixture."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text(pymupdf.Point(50, 50), text)
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_loader_single_page(tmp_path: Path):
    pdf_path = tmp_path / "single_page.pdf"
    content = "NIST CSF 2.0 Identity Management Guidance with CVE-2025-0001."
    create_test_pdf(pdf_path, [content])

    loader = PDFLoader()
    doc = loader.load(pdf_path)

    assert doc.total_pages == 1
    assert doc.source_type == "pdf"
    assert len(doc.checksum) == 64
    assert content in doc.full_text
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].char_count == len(content)


def test_pdf_loader_multi_page_provenance(tmp_path: Path):
    pdf_path = tmp_path / "multi_page.pdf"
    p1 = "Page One: Defensive Control Objectives."
    p2 = "Page Two: Incident Response Planning Procedures."
    create_test_pdf(pdf_path, [p1, p2])

    loader = PDFLoader()
    doc = loader.load(pdf_path)

    assert doc.total_pages == 2
    assert doc.pages[0].page_number == 1
    assert p1 in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert p2 in doc.pages[1].text


def test_pdf_loader_blank_pdf_raises_no_extractable_text(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    create_test_pdf(pdf_path, ["", ""])

    loader = PDFLoader()
    with pytest.raises(NoExtractableTextError) as exc_info:
        loader.load(pdf_path)

    assert "scanned or image-only" in str(exc_info.value) or "characters" in str(exc_info.value)


def test_pdf_loader_missing_file_raises_not_found(tmp_path: Path):
    loader = PDFLoader()
    missing = tmp_path / "non_existent.pdf"
    with pytest.raises(DocumentNotFoundError):
        loader.load(missing)


def test_pdf_loader_corrupt_file_raises_extraction_error(tmp_path: Path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a valid pdf header")

    loader = PDFLoader()
    with pytest.raises(DocumentExtractionError):
        loader.load(corrupt)
