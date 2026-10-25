"""
Repository layer for ContextForge storage entities.
===================================================
Provides clean persistence methods over SQLite without ORM bloat.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from contextforge.common.types import (
    AnswerResult,
    Chunk,
    Document,
    DuplicateDocumentError,
    RetrievedChunk,
)


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


class ChunkRepository:
    """Repository for CRUD operations on the `chunks` SQLite table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize ChunkRepository.

        Args:
            conn: Active sqlite3.Connection to ContextForge database.
        """
        self.conn = conn

    def save_chunks(self, chunks: list[Chunk], embedding_model: str | None = None) -> None:
        """Persist a batch of Chunk records using parameterized SQL.

        Args:
            chunks: List of Chunk instances.
            embedding_model: Optional embedding model identifier.
        """
        if not chunks:
            return

        rows = []
        for c in chunks:
            meta_json = json.dumps(c.metadata, ensure_ascii=False)
            rows.append(
                (
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.chunk_profile,
                    c.text,
                    c.token_count,
                    c.content_hash,
                    c.page_start,
                    c.page_end,
                    c.section_heading,
                    embedding_model,
                    meta_json,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO chunks (
                id, document_id, chunk_index, chunk_profile, text, token_count,
                content_hash, page_start, page_end, section_heading,
                embedding_model, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_id = excluded.document_id,
                chunk_index = excluded.chunk_index,
                chunk_profile = excluded.chunk_profile,
                text = excluded.text,
                token_count = excluded.token_count,
                content_hash = excluded.content_hash,
                page_start = excluded.page_start,
                page_end = excluded.page_end,
                section_heading = excluded.section_heading,
                embedding_model = excluded.embedding_model,
                metadata_json = excluded.metadata_json
            """,
            rows,
        )
        self.conn.commit()

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by its primary key ID."""
        cursor = self.conn.execute(
            """
            SELECT id, document_id, chunk_index, chunk_profile, text, token_count,
                   content_hash, page_start, page_end, section_heading, metadata_json
            FROM chunks WHERE id = ?
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve a list of chunks by IDs preserving order."""
        if not chunk_ids:
            return []

        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = self.conn.execute(
            f"""
            SELECT id, document_id, chunk_index, chunk_profile, text, token_count,
                   content_hash, page_start, page_end, section_heading, metadata_json
            FROM chunks WHERE id IN ({placeholders})
            """,
            chunk_ids,
        )
        id_map = {row["id"]: self._row_to_chunk(row) for row in cursor.fetchall()}
        return [id_map[cid] for cid in chunk_ids if cid in id_map]

    def list_by_profile(self, chunk_profile: str) -> list[Chunk]:
        """List all chunks for a given chunking profile ordered by document and index."""
        cursor = self.conn.execute(
            """
            SELECT id, document_id, chunk_index, chunk_profile, text, token_count,
                   content_hash, page_start, page_end, section_heading, metadata_json
            FROM chunks WHERE chunk_profile = ?
            ORDER BY document_id, chunk_index ASC
            """,
            (chunk_profile,),
        )
        return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def count_by_profile(self, chunk_profile: str) -> int:
        """Count total chunks for a profile."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_profile = ?",
            (chunk_profile,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def delete_by_profile(self, chunk_profile: str) -> int:
        """Delete all chunks for a profile to allow clean rebuilding."""
        cursor = self.conn.execute(
            "DELETE FROM chunks WHERE chunk_profile = ?",
            (chunk_profile,),
        )
        self.conn.commit()
        return cursor.rowcount

    def delete_by_document(self, document_id: str, chunk_profile: str | None = None) -> int:
        """Delete chunks belonging to a document."""
        if chunk_profile:
            cursor = self.conn.execute(
                "DELETE FROM chunks WHERE document_id = ? AND chunk_profile = ?",
                (document_id, chunk_profile),
            )
        else:
            cursor = self.conn.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
        self.conn.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        metadata: dict[str, Any] = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}

        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            chunk_profile=row["chunk_profile"],
            text=row["text"],
            token_count=row["token_count"],
            content_hash=row["content_hash"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_heading=row["section_heading"],
            metadata=metadata,
        )


class QueryResultRepository:
    """Repository for persisting queries, responses, and retrieval events."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize QueryResultRepository.

        Args:
            conn: Active sqlite3.Connection to ContextForge database.
        """
        self.conn = conn

    def save_query_and_result(
        self,
        question: str,
        result: AnswerResult,
        query_id: str | None = None,
    ) -> tuple[str, str]:
        """Atomically persist a query, its generated AnswerResult, and RAG retrieval events.

        Args:
            question: User question string.
            result: AnswerResult returned from Base, RAG, or Fine-tuned pipeline.
            query_id: Optional custom query ID. If None, generated automatically.

        Returns:
            Tuple of (query_id, response_id).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        q_id = query_id or f"q_{uuid.uuid4().hex[:12]}"
        resp_id = f"resp_{uuid.uuid4().hex[:12]}"

        prompt_version = result.metadata.get("prompt_version")
        meta_to_store = {k: v for k, v in result.metadata.items() if k != "prompt_version"}
        metadata_json = json.dumps(meta_to_store, ensure_ascii=False)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO queries (id, question, created_at)
            VALUES (?, ?, ?)
            """,
            (q_id, question, now_iso),
        )

        cursor.execute(
            """
            INSERT INTO responses (
                id, query_id, pipeline, answer, model_id, latency_ms,
                prompt_version, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resp_id,
                q_id,
                result.pipeline,
                result.answer,
                result.model_id,
                result.latency_ms,
                prompt_version,
                now_iso,
                metadata_json,
            ),
        )

        if result.sources:
            for source in result.sources:
                event_id = f"ret_{uuid.uuid4().hex[:12]}"
                src_meta_json = json.dumps(source.metadata, ensure_ascii=False)
                cursor.execute(
                    """
                    INSERT INTO retrieval_events (
                        id, response_id, chunk_id, document_id, source_label, rank, score, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        resp_id,
                        source.chunk_id,
                        source.document_id,
                        source.source_label,
                        source.rank,
                        source.score,
                        src_meta_json,
                    ),
                )

        self.conn.commit()
        return q_id, resp_id

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        """Fetch a response by ID with query information."""
        cursor = self.conn.execute(
            """
            SELECT r.id, r.query_id, q.question, r.pipeline, r.answer, r.model_id,
                   r.latency_ms, r.prompt_version, r.created_at, r.metadata_json
            FROM responses r
            JOIN queries q ON r.query_id = q.id
            WHERE r.id = ?
            """,
            (response_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}

        return {
            "id": row["id"],
            "query_id": row["query_id"],
            "question": row["question"],
            "pipeline": row["pipeline"],
            "answer": row["answer"],
            "model_id": row["model_id"],
            "latency_ms": row["latency_ms"],
            "prompt_version": row["prompt_version"],
            "created_at": row["created_at"],
            "metadata": meta,
        }

    def get_retrieval_events(self, response_id: str) -> list[dict[str, Any]]:
        """Fetch all retrieval events associated with a response, ordered by rank."""
        cursor = self.conn.execute(
            """
            SELECT id, response_id, chunk_id, document_id, source_label, rank, score, metadata_json
            FROM retrieval_events
            WHERE response_id = ?
            ORDER BY rank ASC
            """,
            (response_id,),
        )
        results = []
        for row in cursor.fetchall():
            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    meta = {}
            results.append(
                {
                    "id": row["id"],
                    "response_id": row["response_id"],
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "source_label": row["source_label"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "metadata": meta,
                }
            )
        return results

    def list_recent_queries(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recently executed queries joined with their response."""
        cursor = self.conn.execute(
            """
            SELECT q.id AS query_id, q.question, q.created_at,
                   r.id AS response_id, r.pipeline, r.model_id, r.latency_ms
            FROM queries q
            LEFT JOIN responses r ON q.id = r.query_id
            ORDER BY q.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_queries(self) -> int:
        """Count total queries in the database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM queries")
        row = cursor.fetchone()
        return row[0] if row else 0

    def count_responses(self) -> int:
        """Count total responses in the database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM responses")
        row = cursor.fetchone()
        return row[0] if row else 0

