"""Storage package for ContextForge."""

from contextforge.storage.db import get_connection, init_db
from contextforge.storage.repositories import DocumentRepository

__all__ = ["DocumentRepository", "get_connection", "init_db"]
