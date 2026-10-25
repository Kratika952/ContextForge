"""Pydantic request and response schemas for ContextForge FastAPI backend.
========================================================================
Phase 10 contract matching Sections 6.4 and 8 of IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Incoming user question request schema."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User question text to be answered by the requested pipeline.",
        examples=["How is Broken Access Control prevented in OWASP Top 10:2025?"],
    )
    pipeline: Literal["base", "rag", "finetuned"] = Field(
        default="rag",
        description="Target inference pipeline ('base', 'rag', or 'finetuned').",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        """Reject empty or whitespace-only questions."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace-only.")
        return cleaned


class SourceChunkResponse(BaseModel):
    """Schema representing an individual retrieved context chunk in RAG responses."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    source_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Standardized response schema across Base, RAG, and Fine-tuned pipelines."""

    question: str
    pipeline: Literal["base", "rag", "finetuned"]
    answer: str
    model_id: str
    latency_ms: float
    sources: list[SourceChunkResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Lightweight operational health response schema."""

    status: str = "ok"
    service: str = "contextforge"
    phase: str = "api"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    active_pipeline: str | None = None
    database_connected: bool = True
