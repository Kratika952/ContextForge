"""Unit tests for ContextForge FastAPI backend and persistence.
============================================================
Phase 10: Tests API routing, request validation, pipeline management,
and SQLite persistence without loading real causal language models.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from contextforge.api.main import app, get_pipeline_manager, get_repository
from contextforge.api.pipeline_manager import PipelineManager
from contextforge.common.types import (
    AnswerResult,
    InvalidQuestionError,
    ModelLoadError,
    RetrievedChunk,
)
from contextforge.storage.db import init_db
from contextforge.storage.repositories import QueryResultRepository


@pytest.fixture
def mock_db(tmp_path):
    """Temporary file-backed SQLite database for cross-thread FastAPI test isolation."""
    db_path = tmp_path / "test_contextforge.db"
    conn = init_db(db_path, check_same_thread=False)
    yield conn
    conn.close()



@pytest.fixture
def repo(mock_db: sqlite3.Connection):
    """QueryResultRepository connected to in-memory database."""
    return QueryResultRepository(mock_db)


@pytest.fixture
def mock_pipeline_manager():
    """PipelineManager with mocked pipelines for Base, RAG, and Fine-Tuned."""
    manager = PipelineManager(device="cpu", dtype="float32")

    mock_base = MagicMock()
    mock_base.generate.return_value = AnswerResult(
        question="What is MFA?",
        pipeline="base",
        answer="Base answer about MFA.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=120.0,
        metadata={"prompt_version": "base_v1", "input_tokens": 10, "output_tokens": 8},
    )

    mock_rag = MagicMock()
    mock_rag.generate.return_value = AnswerResult(
        question="What is MFA?",
        pipeline="rag",
        answer="RAG answer about MFA from [SOURCE_1].",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=250.0,
        sources=[
            RetrievedChunk(
                chunk_id="chunk_001",
                document_id="cisa-cpg-2.0",
                text="MFA requires multiple factors.",
                score=0.88,
                rank=1,
                source_label="[SOURCE_1]",
                metadata={"page_start": 3},
            )
        ],
        metadata={
            "prompt_version": "rag_v1",
            "input_tokens": 500,
            "output_tokens": 15,
            "context_tokens": 450,
            "retrieval_latency_ms": 35.0,
        },
    )

    mock_finetuned = MagicMock()
    mock_finetuned.generate.return_value = AnswerResult(
        question="What is MFA?",
        pipeline="finetuned",
        answer="Fine-tuned answer about MFA.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=90.0,
        metadata={"prompt_version": "base_v1", "input_tokens": 10, "output_tokens": 6},
    )

    manager.set_custom_pipeline("base", mock_base)
    manager._factories["base"] = lambda: mock_base
    manager._factories["rag"] = lambda: mock_rag
    manager._factories["finetuned"] = lambda: mock_finetuned

    return manager


@pytest.fixture
def client(mock_pipeline_manager: PipelineManager, repo: QueryResultRepository):
    """FastAPI TestClient with overridden dependencies."""
    app.dependency_overrides[get_pipeline_manager] = lambda: mock_pipeline_manager
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ── Health Endpoint ──────────────────────────────────────────────────────────


def test_health_endpoint(client: TestClient):
    """GET /health must return 200 with service metadata and without loading models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "contextforge"
    assert data["phase"] == "api"
    assert data["database_connected"] is True


# ── Ask Endpoint: Successful Pipeline Execution ──────────────────────────────


def test_ask_base_pipeline(client: TestClient, repo: QueryResultRepository):
    """POST /ask with pipeline='base' returns valid response without sources and persists to DB."""
    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "base"})
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "base"
    assert data["answer"] == "Base answer about MFA."
    assert data["sources"] == []
    assert data["latency_ms"] == 120.0

    # Verify persistence
    assert repo.count_queries() == 1
    assert repo.count_responses() == 1
    recent = repo.list_recent_queries()
    assert recent[0]["pipeline"] == "base"


def test_ask_rag_pipeline(client: TestClient, repo: QueryResultRepository):
    """POST /ask with pipeline='rag' returns sources and persists retrieval events."""
    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "rag"})
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "rag"
    assert "from [SOURCE_1]" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "chunk_001"
    assert data["sources"][0]["source_label"] == "[SOURCE_1]"
    assert data["sources"][0]["rank"] == 1

    # Verify SQLite persistence including retrieval_events table
    recent = repo.list_recent_queries()
    assert len(recent) == 1
    resp_id = recent[0]["response_id"]
    events = repo.get_retrieval_events(resp_id)
    assert len(events) == 1
    assert events[0]["chunk_id"] == "chunk_001"
    assert events[0]["document_id"] == "cisa-cpg-2.0"
    assert events[0]["score"] == 0.88


def test_ask_finetuned_pipeline(client: TestClient, repo: QueryResultRepository):
    """POST /ask with pipeline='finetuned' returns valid concise response without sources."""
    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "finetuned"})
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "finetuned"
    assert data["answer"] == "Fine-tuned answer about MFA."
    assert data["sources"] == []
    assert repo.count_responses() == 1


# ── Request Validation & Error Handling ─────────────────────────────────────


def test_ask_invalid_pipeline_rejected(client: TestClient):
    """POST /ask with an unsupported pipeline name returns 422 validation error."""
    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "unknown_pipe"})
    assert response.status_code == 422
    assert "Input should be 'base', 'rag' or 'finetuned'" in response.text


def test_ask_empty_question_rejected(client: TestClient):
    """POST /ask with empty or whitespace-only question returns 422 validation error."""
    # Empty string
    res1 = client.post("/ask", json={"question": "", "pipeline": "rag"})
    assert res1.status_code == 422

    # Whitespace only
    res2 = client.post("/ask", json={"question": "   \n\t  ", "pipeline": "rag"})
    assert res2.status_code == 422
    assert "cannot be empty" in res2.text


def test_ask_model_load_failure(client: TestClient, mock_pipeline_manager: PipelineManager):
    """POST /ask returns 503 Service Unavailable when model loading fails."""
    mock_pipeline_manager._factories["finetuned"] = MagicMock(
        side_effect=ModelLoadError("Adapter weights corrupted")
    )
    mock_pipeline_manager.unload_current()

    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "finetuned"})
    assert response.status_code == 503
    assert "unavailable" in response.text.lower()


def test_ask_inference_exception_returns_500(client: TestClient, mock_pipeline_manager: PipelineManager):
    """POST /ask returns 500 when pipeline generation raises an unexpected exception."""
    failing_base = MagicMock()
    failing_base.generate.side_effect = RuntimeError("CUDA driver crash")
    mock_pipeline_manager.set_custom_pipeline("base", failing_base)

    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "base"})
    assert response.status_code == 500
    assert "CUDA driver crash" in response.json()["detail"]


# ── Lifecycle & Pipeline Manager Isolation ───────────────────────────────────


def test_pipeline_manager_reuses_current_pipeline():
    """PipelineManager reuses the currently resident pipeline without recreating it."""
    manager = PipelineManager(device="cpu")
    mock_pipe = MagicMock()
    create_count = 0

    def factory():
        nonlocal create_count
        create_count += 1
        return mock_pipe

    manager.set_factory("base", factory)

    p1 = manager.get_pipeline("base")
    p2 = manager.get_pipeline("base")

    assert p1 is mock_pipe
    assert p2 is mock_pipe
    assert create_count == 1  # Only initialized once


def test_pipeline_manager_unloads_when_switching():
    """PipelineManager safely unloads previous pipeline when switching to a different one."""
    manager = PipelineManager(device="cpu")
    mock_base = MagicMock()
    mock_rag = MagicMock()

    manager.set_factory("base", lambda: mock_base)
    manager.set_factory("rag", lambda: mock_rag)

    p_base = manager.get_pipeline("base")
    assert manager.active_pipeline_name == "base"
    assert p_base is mock_base

    # Switch to RAG
    p_rag = manager.get_pipeline("rag")
    assert manager.active_pipeline_name == "rag"
    assert p_rag is mock_rag


def test_no_silent_pipeline_fallback(client: TestClient, mock_pipeline_manager: PipelineManager):
    """If a requested pipeline fails, system must not silently fallback to Base."""
    mock_pipeline_manager._factories["finetuned"] = MagicMock(
        side_effect=ModelLoadError("Missing adapter")
    )
    mock_pipeline_manager.unload_current()

    response = client.post("/ask", json={"question": "What is MFA?", "pipeline": "finetuned"})
    assert response.status_code == 503
    # Verify that base was not called in response
    assert "Missing adapter" in response.text
