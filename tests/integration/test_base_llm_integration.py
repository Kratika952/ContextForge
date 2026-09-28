"""
Real Model Integration Test for ContextForge Base LLM Pipeline.
==============================================================
Validates that Qwen/Qwen2.5-1.5B-Instruct loads, formats prompts via chat template,
and performs deterministic causal text generation returning AnswerResult.
"""

from __future__ import annotations

import pytest

from contextforge.common.types import AnswerResult
from contextforge.pipelines.base import BasePipeline


@pytest.mark.integration
def test_real_qwen_base_pipeline_inference():
    """Verify real Qwen2.5-1.5B-Instruct model loading and deterministic inference."""
    pipeline = BasePipeline(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto",
        dtype="auto",
        max_new_tokens=64,
        prompt_version="base_v1",
    )

    # Validate model loading
    assert pipeline.model_loader.model is not None
    assert pipeline.model_loader.tokenizer is not None
    assert pipeline.model_loader.is_loaded is True

    question = "What is the primary difference between authentication and authorization?"
    result = pipeline.generate(question)

    assert isinstance(result, AnswerResult)
    assert result.pipeline == "base"
    assert result.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.sources == []
    assert len(result.answer) > 20
    assert result.latency_ms > 0.0
    assert result.metadata["input_token_count"] > 0
    assert result.metadata["output_token_count"] > 0
    assert result.metadata["device"] in ("cpu", "cuda")
