"""Unit tests for the IngestionService."""

from pathlib import Path
import pymupdf
import pytest

from contextforge.common.types import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    UnsupportedDocumentTypeError,
)
from contextforge.ingestion.service import IngestionService


def create_sample_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(pymupdf.Point(50, 50), text)
    doc.save(str(path))
    doc.close()
    return path


def test_service_ingests_pdf(tmp_path: Path):
    db_path = tmp_path / "test.db"
    pdf_path = create_sample_pdf(tmp_path / "sample.pdf", "Sample NIST Guidance Content.")

    with IngestionService(db_path=db_path) as service:
        res = service.ingest_file(pdf_path, title="Sample PDF")
        assert res.status == "INGESTED"
        assert res.document.title == "Sample PDF"
        assert res.document.source_type == "pdf"
        assert service.repo.count() == 1

        # Second ingestion: duplicate skipped
        dup_res = service.ingest_file(pdf_path, on_duplicate="skip")
        assert dup_res.status == "ALREADY_INGESTED"
        assert service.repo.count() == 1


def test_service_duplicate_raise_error(tmp_path: Path):
    db_path = tmp_path / "test.db"
    pdf_path = create_sample_pdf(tmp_path / "sample.pdf", "Content to be duplicated.")

    with IngestionService(db_path=db_path) as service:
        service.ingest_file(pdf_path)
        with pytest.raises(DuplicateDocumentError):
            service.ingest_file(pdf_path, on_duplicate="raise")


def test_service_ingests_markdown(tmp_path: Path):
    db_path = tmp_path / "test.db"
    md_path = tmp_path / "A01_Control.md"
    md_path.write_text("# A01: Broken Access Control\n\nPolicy guidance text.", encoding="utf-8")

    with IngestionService(db_path=db_path) as service:
        res = service.ingest_file(md_path)
        assert res.status == "INGESTED"
        assert res.document.source_type == "markdown"
        assert res.document.title == "A01: Broken Access Control"
        assert service.repo.count() == 1


def test_service_ingests_directory(tmp_path: Path):
    db_path = tmp_path / "test.db"
    docs_dir = tmp_path / "corpus"
    docs_dir.mkdir()

    create_sample_pdf(docs_dir / "doc1.pdf", "First PDF document text.")
    (docs_dir / "doc2.md").write_text("# Doc 2\n\nMarkdown document text.", encoding="utf-8")

    with IngestionService(db_path=db_path) as service:
        results = service.ingest_directory(docs_dir)
        assert len(results) == 2
        assert all(r.status == "INGESTED" for r in results)
        assert service.repo.count() == 2


def test_service_rejects_unsupported_extension(tmp_path: Path):
    db_path = tmp_path / "test.db"
    doc_path = tmp_path / "document.docx"
    doc_path.write_text("dummy docx content", encoding="utf-8")

    with IngestionService(db_path=db_path) as service:
        with pytest.raises(UnsupportedDocumentTypeError):
            service.ingest_file(doc_path)


def test_service_missing_file_raises_error(tmp_path: Path):
    db_path = tmp_path / "test.db"
    with IngestionService(db_path=db_path) as service:
        with pytest.raises(DocumentNotFoundError):
            service.ingest_file(tmp_path / "missing.pdf")
