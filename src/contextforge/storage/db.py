"""
Database connection and initialization for ContextForge SQLite database.
========================================================================
Uses Python's standard library sqlite3 with WAL mode and foreign key enforcement.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from contextforge.common.config import get_db_path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection to the ContextForge SQLite database.

    Args:
        db_path: Path to database file, or ':memory:', or None (uses config default).

    Returns:
        sqlite3.Connection with row_factory=sqlite3.Row and foreign keys enabled.
    """
    if db_path is None:
        target = get_db_path()
    elif str(db_path) == ":memory:":
        target = ":memory:"
    else:
        target = Path(db_path).resolve()

    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target))
    else:
        conn = sqlite3.connect(target)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if str(target) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")

    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply incremental column migrations if tables exist with older schemas."""
    # Check chunks table columns
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
    if cursor.fetchone():
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "chunk_profile" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN chunk_profile TEXT NOT NULL DEFAULT 'medium';")
        if "section_heading" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN section_heading TEXT;")
        if "metadata_json" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';")


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Initialize database tables according to schema.sql.

    Args:
        db_path: Path to database file, or ':memory:', or None.

    Returns:
        Initialized sqlite3.Connection.
    """
    conn = get_connection(db_path)
    _migrate_schema(conn)
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    return conn
