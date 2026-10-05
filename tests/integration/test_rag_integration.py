"""
Real Integration Test for ContextForge RAG Pipeline.
====================================================
Validates end-to-end RAG inference using real BGE query embeddings,
real FAISS dense retrieval against medium index, real SQLite chunk metadata,
and real deterministic Qwen2.5 generation.
"""

from __future__ import annotations

import pytest

from contextforge.common.types import AnswerResult
from contextforge.pipelines.rag import RagPipeline


@pytest.mark.integration
def test_real_rag_pipeline_end_to_end_inference():
    """Verify live RAG inference with real BGE embeddings, FAISS retrieval, and Qwen generation."""
    pipeline = RagPipeline(
        chunk_profile="medium",
        top_k=2,
        max_new_tokens=64,
        prompt_version="rag_v1",
    )

    try:
        assert pipeline.retriever.store.is_built is True
        assert pipeline.retriever.store.total_vectors > 0
        assert pipeline.model_loader.model is not None

        question = "What is the purpose of multi-factor authentication?"
        result = pipeline.generate(question)

        assert isinstance(result, AnswerResult)
        assert result.pipeline == "rag"
        assert result.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        assert len(result.sources) == 2
        assert len(result.answer) > 10

        meta = result.metadata
        assert meta["prompt_version"] == "rag_v1"
        assert meta["chunk_profile"] == "medium"
        assert meta["retrieved_count"] == 2
        assert meta["included_count"] > 0
        assert meta["context_token_count"] > 0
        assert meta["retrieval_latency_ms"] > 0.0
        assert meta["generation_latency_ms"] > 0.0
        assert meta["total_latency_ms"] > 0.0
        assert isinstance(meta["citation_labels_used"], list)
        assert isinstance(meta["invalid_citations"], list)
    finally:
        pipeline.close()
