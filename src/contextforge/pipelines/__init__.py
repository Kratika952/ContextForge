"""
ContextForge Pipelines Package.
===============================
Standardized pipeline interfaces for Base, RAG, and fine-tuned model inference.
"""

from __future__ import annotations

from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.rag import RagPipeline

__all__ = [
    "BasePipeline",
    "RagPipeline",
]

