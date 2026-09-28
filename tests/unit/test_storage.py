"""Unit tests for SQLite storage layer and DocumentRepository."""

from pathlib import Path
import sqlite3
import pytest

from contextforge.common.types import Document, DuplicateDocumentError
from contextforge.storage.db import init_db
from contextforge.storage.repositories import DocumentRepository


def test_db_initialization(tmp_path: Path):
    db_file = tmp_path / "test.db"
    conn = init_db(db_file)
    try:
        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        assert "documents" in tables
        assert "chunks" in tables
    finally:
        conn.close()


def test_repository_save_and_lookup(tmp_path: Path):
    conn = init_db(tmp_path / "test.db")
    try:
        repo = DocumentRepository(conn)
        doc = Document(
            id="nist-csf-2.0",
            title="NIST CSF 2.0",
            source_path="data/raw/nist-csf-2.0.pdf",
            checksum="3c31f46fee98cac0c4323453e5109291a213b4de7fef8c058af9bf67f717433c",
            source_type="pdf",
            status="INGESTED",
            metadata={"pages": 32},
        )
        repo.save(doc)

        assert repo.count() == 1

        # Lookup by ID
        fetched = repo.get_by_id("nist-csf-2.0")
        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.title == doc.title
        assert fetched.source_type == "pdf"
        assert fetched.metadata["pages"] == 32

        # Lookup by Checksum
        by_hash = repo.get_by_checksum(doc.checksum)
        assert by_hash is not None
        assert by_hash.id == doc.id
    finally:
        conn.close()


def test_repository_prevents_duplicate_checksum(tmp_path: Path):
    conn = init_db(tmp_path / "test.db")
    try:
        repo = DocumentRepository(conn)
        doc1 = Document(
            id="doc-one",
            title="Document One",
            source_path="data/raw/doc1.pdf",
            checksum="aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff66667777888899990000",
            source_type="pdf",
        )
        repo.save(doc1)

        # Attempt to insert a different document ID with identical checksum
        doc2 = Document(
            id="doc-two",
            title="Document Two",
            source_path="data/raw/doc2.pdf",
            checksum="aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff66667777888899990000",
            source_type="pdf",
        )
        with pytest.raises(DuplicateDocumentError):
            repo.save(doc2)

        assert repo.count() == 1
    finally:
        conn.close()


def test_repository_delete(tmp_path: Path):
    conn = init_db(tmp_path / "test.db")
    try:
        repo = DocumentRepository(conn)
        doc = Document(
            id="to-delete",
            title="Delete Me",
            source_path="path.pdf",
            checksum="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        )
        repo.save(doc)
        assert repo.count() == 1

        deleted = repo.delete("to-delete")
        assert deleted is True
        assert repo.count() == 0
    finally:
        conn.close()
