"""Ingestion package for ContextForge."""

from contextforge.ingestion.cleaner import clean_text
from contextforge.ingestion.markdown_loader import MarkdownLoader
from contextforge.ingestion.pdf_loader import PDFLoader, compute_sha256
from contextforge.ingestion.service import IngestionResult, IngestionService

__all__ = [
    "IngestionResult",
    "IngestionService",
    "MarkdownLoader",
    "PDFLoader",
    "clean_text",
    "compute_sha256",
]
