"""
Repository layer for ContextForge storage entities.
===================================================
Provides clean persistence methods over SQLite without ORM bloat.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from contextforge.common.types import Document, DuplicateDocumentError


class DocumentRepository:
    """Repository for CRUD operations on the `documents` SQLite table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize DocumentRepository.

        Args:
            conn: Active sqlite3.Connection to ContextForge database.
        """
        self.conn = conn

    def save(self, doc: Document) -> None:
        """Persist a Document record.

        Args:
            doc: Document instance to insert or update.

        Raises:
            DuplicateDocumentError: If a different document with the same checksum already exists.
        """
        existing = self.get_by_checksum(doc.checksum)
        if existing and existing.id != doc.id:
            raise DuplicateDocumentError(
                f"Document with checksum '{doc.checksum}' already exists "
                f"(id: '{existing.id}', title: '{existing.title}')."
            )

        metadata_str = json.dumps(doc.metadata, ensure_ascii=False)
        try:
            self.conn.execute(
                """
                INSERT INTO documents (
                    id, title, source_path, checksum, source_type, status, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    source_path = excluded.source_path,
                    checksum = excluded.checksum,
                    source_type = excluded.source_type,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    doc.id,
                    doc.title,
                    doc.source_path,
                    doc.checksum,
                    doc.source_type,
                    doc.status,
                    doc.created_at,
                    metadata_str,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: documents.checksum" in str(exc):
                raise DuplicateDocumentError(
                    f"Document with checksum '{doc.checksum}' already exists."
                ) from exc
            raise

    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a document by its primary key ID."""
        cursor = self.conn.execute(
            """
            SELECT id, title, source_path, checksum, source_type, status, created_at, metadata_json
            FROM documents WHERE id = ?
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def get_by_checksum(self, checksum: str) -> Document | None:
        """Retrieve a document by its SHA-256 checksum."""
        cursor = self.conn.execute(
            """
            SELECT id, title, source_path, checksum, source_type, status, created_at, metadata_json
            FROM documents WHERE checksum = ?
            """,
            (checksum.lower(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def list_all(self) -> list[Document]:
        """List all documents ordered by created_at."""
        cursor = self.conn.execute(
            """
            SELECT id, title, source_path, checksum, source_type, status, created_at, metadata_json
            FROM documents ORDER BY created_at ASC
            """
        )
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Return total number of documents in repository."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        row = cursor.fetchone()
        return row[0] if row else 0

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted, False if not found."""
        cursor = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        metadata: dict[str, Any] = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}

        # Safely extract source_type if present in row
        source_type = "pdf"
        try:
            source_type = row["source_type"]
        except (IndexError, KeyError):
            pass

        return Document(
            id=row["id"],
            title=row["title"],
            source_path=row["source_path"],
            checksum=row["checksum"],
            source_type=source_type,
            status=row["status"],
            created_at=row["created_at"],
            metadata=metadata,
        )
