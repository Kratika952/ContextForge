"""
ContextForge Retrieval Package.
===============================
Dense semantic retrieval using Sentence Transformers (BGE-small-en-v1.5)
and FAISS flat inner-product vector indexing.
"""

from __future__ import annotations

from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.retrieval.retriever import DenseRetriever

__all__ = [
    "BGEEmbedder",
    "FAISSStore",
    "DenseRetriever",
]
