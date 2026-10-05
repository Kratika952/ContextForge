"""
Unit Tests for ContextForge RAG Pipeline.
=========================================
Tests question validation, retriever orchestration, context formatting,
shared generator reuse, citation validation, and base isolation regression with mocked boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from contextforge.common.types import (
    AnswerResult,
    InvalidQuestionError,
    RetrievedChunk,
)
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.context_formatter import ContextFormatter
from contextforge.llm.generation import GenerationResult, TextGenerator
from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.rag import RagPipeline
from contextforge.retrieval.retriever import DenseRetriever


def _create_mock_chunk(chunk_id: str, doc_id: str, text: str, score: float = 0.88, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        score=score,
        rank=rank,
        source_label=f"[{doc_id}-CHK{rank:02d}]",
        metadata={"token_count": len(text.split())},
    )


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=DenseRetriever)
    retriever.embedder = MagicMock()
    retriever.embedder.model_name = "BAAI/bge-small-en-v1.5"
    retriever.store = MagicMock()
    retriever.store.chunk_profile = "medium"
    retriever.retrieve.return_value = [
        _create_mock_chunk("c1", "nist-csf-20", "The Govern function establishes cybersecurity risk governance.", score=0.91, rank=1),
        _create_mock_chunk("c2", "owasp-a01", "Broken access control occurs when users can act outside permissions.", score=0.87, rank=2),
    ]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock(spec=TextGenerator)
    generator.generate.return_value = GenerationResult(
        answer="According to [SOURCE_1], the Govern function sets risk strategy. [SOURCE_2] explains access control.",
        input_token_count=180,
        output_token_count=42,
        raw_prompt="<mock_prompt>",
    )
    return generator


@pytest.fixture
def mock_model_loader():
    loader = MagicMock(spec=BaseModelLoader)
    loader.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loader.device = "cpu"
    loader.dtype = "bfloat16"
    loader.tokenizer = None
    return loader


def test_rag_pipeline_question_validation(mock_retriever, mock_generator, mock_model_loader):
    pipeline = RagPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        model_loader=mock_model_loader,
    )

    with pytest.raises(InvalidQuestionError, match="cannot be empty"):
        pipeline.generate("")

    with pytest.raises(InvalidQuestionError, match="cannot be empty"):
        pipeline.generate("   \n\t  ")

    with pytest.raises(InvalidQuestionError, match="must be a string"):
        pipeline.generate(None)  # type: ignore

    with pytest.raises(InvalidQuestionError, match="exceeds maximum allowed length"):
        pipeline.generate("a" * 4001)


def test_rag_pipeline_orchestration_and_result_contract(mock_retriever, mock_generator, mock_model_loader):
    pipeline = RagPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        model_loader=mock_model_loader,
        top_k=5,
        max_context_tokens=1000,
        prompt_version="rag_v1",
    )

    question = "What is the Govern function in NIST CSF 2.0?"
    result = pipeline.generate(question)

    # 1. Verify retriever was called once with question and top_k
    mock_retriever.retrieve.assert_called_once_with(question, top_k=5)

    # 2. Verify generator was called with formatted messages
    mock_generator.generate.assert_called_once()
    called_kwargs = mock_generator.generate.call_args.kwargs
    assert "messages" in called_kwargs
    messages = called_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "ContextForge" in messages[0]["content"]
    assert "only the provided evidence" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "<BEGIN_EVIDENCE>" in messages[1]["content"]
    assert "<END_EVIDENCE>" in messages[1]["content"]
    assert "QUESTION:\n" + question in messages[1]["content"]

    # 3. Verify AnswerResult contract
    assert isinstance(result, AnswerResult)
    assert result.pipeline == "rag"
    assert result.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.latency_ms > 0.0
    assert len(result.sources) == 2
    assert result.sources[0].chunk_id == "c1"
    assert result.sources[1].chunk_id == "c2"

    # 4. Verify telemetry metadata
    meta = result.metadata
    assert meta["prompt_version"] == "rag_v1"
    assert meta["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert meta["chunk_profile"] == "medium"
    assert meta["top_k_requested"] == 5
    assert meta["retrieved_count"] == 2
    assert meta["included_count"] == 2
    assert meta["context_token_count"] > 0
    assert meta["input_token_count"] == 180
    assert meta["output_token_count"] == 42
    assert meta["retrieval_latency_ms"] >= 0.0
    assert meta["generation_latency_ms"] >= 0.0
    assert meta["total_latency_ms"] >= 0.0
    assert meta["citation_labels_used"] == ["SOURCE_1", "SOURCE_2"]
    assert meta["invalid_citations"] == []


def test_rag_pipeline_flags_invalid_citation(mock_retriever, mock_generator, mock_model_loader):
    # Model generates citation to SOURCE_99 which was never provided
    mock_generator.generate.return_value = GenerationResult(
        answer="According to [SOURCE_1] and fabricated [SOURCE_99], security is crucial.",
        input_token_count=100,
        output_token_count=30,
        raw_prompt="<mock_prompt>",
    )

    pipeline = RagPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        model_loader=mock_model_loader,
    )

    result = pipeline.generate("Explain security governance.")

    assert result.metadata["citation_labels_used"] == ["SOURCE_1", "SOURCE_99"]
    assert result.metadata["invalid_citations"] == ["SOURCE_99"]
    # Sources list is STILL the application-owned list of 2 retrieved chunks
    assert len(result.sources) == 2


def test_base_pipeline_isolation_regression():
    """Verify that BasePipeline remains 100% free of FAISS, retriever, and corpus dependencies."""
    mock_loader = MagicMock(spec=BaseModelLoader)
    mock_loader.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    mock_gen = MagicMock(spec=TextGenerator)
    mock_gen.generate.return_value = GenerationResult(
        answer="Base answer without sources.",
        input_token_count=50,
        output_token_count=20,
        raw_prompt="<mock_prompt>",
    )

    base = BasePipeline(model_loader=mock_loader, generator=mock_gen)
    res = base.generate("What is authorization?")

    assert res.pipeline == "base"
    assert res.sources == []
    assert not hasattr(base, "retriever")
    assert not hasattr(base, "store")
