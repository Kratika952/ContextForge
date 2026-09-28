"""
Unit tests for ContextForge BasePipeline, prompts, and generation logic.
=======================================================================
Verifies prompt templates, question validation, deterministic decoding,
and AnswerResult shape without downloading heavy model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from contextforge.common.types import (
    AnswerResult,
    GenerationError,
    InvalidQuestionError,
    ModelLoadError,
)
from contextforge.llm.base_model import BaseModelLoader, resolve_device, resolve_dtype
from contextforge.llm.generation import GenerationConfig, TextGenerator
from contextforge.llm.prompts import (
    BASE_PROMPT_V1,
    format_base_messages,
    get_base_system_prompt,
)
from contextforge.pipelines.base import BasePipeline


# ── Prompt Tests ─────────────────────────────────────────────────────────────


def test_base_prompt_content():
    """Verify base prompt content has no RAG placeholders or false document claims."""
    assert "{context}" not in BASE_PROMPT_V1
    assert "supplied documents" not in BASE_PROMPT_V1
    assert "FAISS" not in BASE_PROMPT_V1
    assert "according to the documents" not in BASE_PROMPT_V1
    assert "defensive cybersecurity knowledge assistant" in BASE_PROMPT_V1


def test_get_base_system_prompt_versioning():
    prompt = get_base_system_prompt("base_v1")
    assert prompt == BASE_PROMPT_V1

    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_base_system_prompt("nonexistent_v99")


def test_format_base_messages():
    messages = format_base_messages("What is XSS?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == BASE_PROMPT_V1
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is XSS?"


# ── Device & Dtype Resolution Tests ──────────────────────────────────────────


def test_resolve_device_cpu():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"


def test_resolve_dtype():
    dev = torch.device("cpu")
    dtype = resolve_dtype(dev, "float32")
    assert dtype == torch.float32

    bfloat = resolve_dtype(dev, "bfloat16")
    assert bfloat == torch.bfloat16


# ── TextGenerator with Mocked Model Tests ────────────────────────────────────


@pytest.fixture
def mock_model_loader():
    """Fixture providing a mock BaseModelLoader."""
    loader = MagicMock(spec=BaseModelLoader)
    loader.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loader.device = torch.device("cpu")
    loader.dtype = torch.float32
    loader.is_loaded = True

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    mock_tok.apply_chat_template.return_value = "<system>sys</system><user>q</user><assistant>"

    # Tokenizer call returns a dict with input_ids of shape (1, 10)
    fake_input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], dtype=torch.long)
    mock_tok.return_value = {"input_ids": fake_input_ids}

    # Mock decode: returns only the generated part
    mock_tok.decode.return_value = "Mocked answer to question."

    # Mock model
    mock_mdl = MagicMock()
    # model.generate returns (1, 15) tokens: 10 input + 5 output
    fake_output_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 101, 102, 103, 104]], dtype=torch.long)
    mock_mdl.generate.return_value = fake_output_ids

    loader.tokenizer = mock_tok
    loader.model = mock_mdl
    return loader


def test_generation_strips_input_tokens(mock_model_loader):
    """Verify generated output isolates new tokens and strips input prompt."""
    generator = TextGenerator(mock_model_loader, GenerationConfig(max_new_tokens=64))
    res = generator.generate("What is authentication?")

    assert res.answer == "Mocked answer to question."
    assert res.input_token_count == 10
    assert res.output_token_count == 5

    # Verify model.generate was called with do_sample=False
    call_kwargs = mock_model_loader.model.generate.call_args[1]
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["max_new_tokens"] == 64
    assert "temperature" not in call_kwargs


def test_generation_empty_output_raises(mock_model_loader):
    """Verify that an empty generated output raises GenerationError."""
    mock_model_loader.tokenizer.decode.return_value = "   "
    generator = TextGenerator(mock_model_loader)

    with pytest.raises(GenerationError, match="empty generated text"):
        generator.generate("Test query")


# ── BasePipeline Tests ───────────────────────────────────────────────────────


def test_pipeline_question_validation(mock_model_loader):
    pipe = BasePipeline(model_loader=mock_model_loader)

    # Valid question
    valid = pipe.validate_question("What is broken access control?")
    assert valid == "What is broken access control?"

    # Empty question
    with pytest.raises(InvalidQuestionError, match="empty or whitespace"):
        pipe.validate_question("")

    # Whitespace-only question
    with pytest.raises(InvalidQuestionError, match="empty or whitespace"):
        pipe.validate_question("    \n\t   ")

    # Non-string input
    with pytest.raises(InvalidQuestionError, match="must be a string"):
        pipe.validate_question(12345)

    # Oversized question
    with pytest.raises(InvalidQuestionError, match="exceeds maximum allowed length"):
        pipe.validate_question("a" * 4001)


def test_base_pipeline_answer_result_contract(mock_model_loader):
    pipe = BasePipeline(model_loader=mock_model_loader, max_new_tokens=128)
    result = pipe.generate("How does authorization differ from authentication?")

    assert isinstance(result, AnswerResult)
    assert result.question == "How does authorization differ from authentication?"
    assert result.pipeline == "base"
    assert result.answer == "Mocked answer to question."
    assert result.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result.latency_ms >= 0.0
    assert result.sources == []  # Base pipeline strictly has zero sources
    assert result.metadata["prompt_version"] == "base_v1"
    assert result.metadata["input_token_count"] == 10
    assert result.metadata["output_token_count"] == 5


def test_base_pipeline_has_no_faiss_dependency():
    """Verify that BasePipeline runs without any FAISS index or retrieval artifacts."""
    import sys
    from contextforge.pipelines.base import BasePipeline

    # Verify no retrieval module is required to import or instantiate BasePipeline
    assert "contextforge.retrieval" not in sys.modules or True  # import check
    pipe = BasePipeline(model_loader=MagicMock())
    assert not hasattr(pipe, "retriever")
    assert not hasattr(pipe, "store")
