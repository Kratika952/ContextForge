"""Unit tests for ContextForge FineTunedPipeline.
=============================================
Verifies question validation, AnswerResult shape, adapter load handling,
and deterministic decoding settings with mocked model/adapter boundaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contextforge.common.types import AnswerResult, InvalidQuestionError
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.generation import GenerationConfig, GenerationResult, TextGenerator
from contextforge.pipelines.finetuned import FineTunedPipeline


@pytest.fixture
def mock_model_loader():
    loader = MagicMock(spec=BaseModelLoader)
    loader.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loader.device = "cpu"
    loader.dtype = "float32"
    loader.model = MagicMock()
    loader.tokenizer = MagicMock()
    return loader


@pytest.fixture
def mock_generator(mock_model_loader):
    gen = MagicMock(spec=TextGenerator)
    gen.model_loader = mock_model_loader
    gen.config = GenerationConfig(max_new_tokens=256, do_sample=False)
    gen.generate.return_value = GenerationResult(
        answer="Fine-tuned model defensive response.",
        input_token_count=45,
        output_token_count=18,
        raw_prompt="System + Question",
    )
    return gen


def test_finetuned_pipeline_validation(mock_model_loader, mock_generator):
    """Verify input validation rules on question strings."""
    pipeline = FineTunedPipeline(
        adapter_path=Path("nonexistent/adapter"),
        model_loader=mock_model_loader,
        generator=mock_generator,
    )

    with pytest.raises(InvalidQuestionError, match="must be a string"):
        pipeline.generate(123)

    with pytest.raises(InvalidQuestionError, match="empty or whitespace-only"):
        pipeline.generate("   ")

    with pytest.raises(InvalidQuestionError, match="exceeds maximum allowed length"):
        pipeline.generate("x" * 4001)


def test_finetuned_pipeline_generate(mock_model_loader, mock_generator):
    """Verify AnswerResult attributes and metadata formatting."""
    pipeline = FineTunedPipeline(
        adapter_path=Path("nonexistent/adapter"),
        model_loader=mock_model_loader,
        generator=mock_generator,
    )

    result = pipeline.generate("What is OWASP A01?")
    assert isinstance(result, AnswerResult)
    assert result.pipeline == "finetuned"
    assert result.model_id == "Qwen/Qwen2.5-1.5B-Instruct+lora"
    assert result.sources == []  # No retrieval sources
    assert result.answer == "Fine-tuned model defensive response."
    assert result.metadata["input_token_count"] == 45
    assert result.metadata["output_token_count"] == 18
    assert result.metadata["adapter_active"] is False  # Nonexistent path fixture


def test_finetuned_pipeline_callable(mock_model_loader, mock_generator):
    """Verify __call__ syntax works equivalently to generate()."""
    pipeline = FineTunedPipeline(
        adapter_path=Path("nonexistent/adapter"),
        model_loader=mock_model_loader,
        generator=mock_generator,
    )
    res = pipeline("What is Broken Access Control?")
    assert res.answer == "Fine-tuned model defensive response."
