"""FastAPI backend application for ContextForge.
==============================================
Exposes Base, RAG, and Fine-Tuned inference endpoints with SQLite persistence.
Phase 10 contract matching Sections 6.4, 7, and 8 of IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from contextforge.api.pipeline_manager import PipelineManager
from contextforge.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceChunkResponse,
)
from contextforge.common.types import (
    IndexNotFoundError,
    InvalidQuestionError,
    ModelLoadError,
    RetrievalError,
)
from contextforge.storage.db import get_connection, init_db
from contextforge.storage.repositories import QueryResultRepository

logger = logging.getLogger("contextforge.api")

# Global default manager and repository instances
_pipeline_manager: PipelineManager | None = None
_repository: QueryResultRepository | None = None


def get_pipeline_manager() -> PipelineManager:
    """Dependency provider for PipelineManager."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


def get_repository() -> QueryResultRepository:
    """Dependency provider for QueryResultRepository."""
    global _repository
    if _repository is None:
        conn = get_connection()
        _repository = QueryResultRepository(conn)
    return _repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown."""
    logger.info("Initializing ContextForge API database schema...")
    try:
        init_db()
        logger.info("ContextForge database initialized.")
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)

    yield

    logger.info("Shutting down ContextForge API...")
    if _pipeline_manager is not None:
        _pipeline_manager.unload_current()


app = FastAPI(
    title="ContextForge API",
    version="0.1.0",
    description="Domain-specific QA backend comparing Base LLM, RAG, and Fine-Tuned models.",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Operational health check",
    tags=["System"],
)
def health_check(
    manager: PipelineManager = Depends(get_pipeline_manager),
    repo: QueryResultRepository = Depends(get_repository),
) -> HealthResponse:
    """Return service health status without triggering heavyweight model loading."""
    db_ok = True
    try:
        repo.conn.execute("SELECT 1").fetchone()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        service="contextforge",
        phase="api",
        model_id=manager.model_id,
        active_pipeline=manager.active_pipeline_name,
        database_connected=db_ok,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Execute question answering inference",
    tags=["Inference"],
    responses={
        400: {"description": "Invalid question input"},
        422: {"description": "Validation error (invalid schema or pipeline name)"},
        502: {"description": "Retrieval component failure"},
        503: {"description": "Model or vector index unavailable"},
        500: {"description": "Internal generation failure"},
    },
)
def ask_question(
    request: AskRequest,
    manager: PipelineManager = Depends(get_pipeline_manager),
    repo: QueryResultRepository = Depends(get_repository),
) -> AskResponse:
    """Execute domain question answering using the selected pipeline.

    Supported pipelines:
      - 'base': Pure causal generation with base LLM.
      - 'rag': Dense retrieval-augmented generation with grounding sources.
      - 'finetuned': Domain-adapted causal generation via PEFT LoRA adapter.
    """
    logger.info("Received /ask request for pipeline '%s' (%d chars)", request.pipeline, len(request.question))

    # Obtain the requested pipeline instance
    try:
        pipeline = manager.get_pipeline(request.pipeline)
    except (IndexNotFoundError, ModelLoadError) as err:
        logger.error("Pipeline initialization dependency error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline '{request.pipeline}' unavailable: {err}",
        ) from err
    except Exception as err:
        logger.error("Failed to obtain pipeline '%s': %s", request.pipeline, err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error loading pipeline '{request.pipeline}': {err}",
        ) from err

    # Execute inference
    try:
        answer_result = pipeline.generate(request.question)
    except InvalidQuestionError as err:
        logger.warning("Rejected invalid question: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except RetrievalError as err:
        logger.error("RAG retrieval failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Evidence retrieval failed: {err}",
        ) from err
    except Exception as err:
        logger.error("Inference execution failed on pipeline '%s': %s", request.pipeline, err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference generation failed: {err}",
        ) from err

    # Persist query and response metadata to SQLite
    try:
        query_id, response_id = repo.save_query_and_result(request.question, answer_result)
        logger.info("Persisted query '%s' and response '%s'", query_id, response_id)
    except Exception as err:
        logger.error("Failed to persist query result to database: %s", err, exc_info=True)
        # We do not fail the client request if database persistence fails, but we record the failure in metadata
        answer_result.metadata["persistence_error"] = str(err)

    # Format response
    formatted_sources = [
        SourceChunkResponse(
            chunk_id=src.chunk_id,
            document_id=src.document_id,
            text=src.text,
            score=src.score,
            rank=src.rank,
            source_label=src.source_label,
            metadata=src.metadata,
        )
        for src in answer_result.sources
    ]

    return AskResponse(
        question=answer_result.question,
        pipeline=answer_result.pipeline,
        answer=answer_result.answer,
        model_id=answer_result.model_id,
        latency_ms=answer_result.latency_ms,
        sources=formatted_sources,
        metadata=answer_result.metadata,
    )
