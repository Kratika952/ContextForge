"""
Base LLM Pipeline for ContextForge.
===================================
Executes baseline domain question answering without retrieval, corpus evidence,
or LoRA adapters. Provides the untouched reference inference path.
"""

from __future__ import annotations

import time
from typing import Any

from contextforge.common.types import AnswerResult, InvalidQuestionError
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.generation import GenerationConfig, TextGenerator

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT_VERSION = "base_v1"
MAX_QUESTION_LENGTH = 4000


class BasePipeline:
    """Baseline LLM inference pipeline without retrieval or adapter augmentation."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        model_loader: BaseModelLoader | None = None,
        generator: TextGenerator | None = None,
    ) -> None:
        """Initialize BasePipeline.

        Args:
            model_id: Model repository ID.
            device: Compute device ('auto', 'cuda', 'cpu').
            dtype: Weight dtype ('auto', 'float32', 'bfloat16', 'float16').
            max_new_tokens: Maximum tokens to generate.
            prompt_version: System prompt template identifier.
            model_loader: Pre-configured BaseModelLoader instance (optional).
            generator: Pre-configured TextGenerator instance (optional).
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.prompt_version = prompt_version

        if model_loader is not None:
            self.model_loader = model_loader
        else:
            self.model_loader = BaseModelLoader(
                model_id=model_id,
                device=device,
                dtype=dtype,
            )

        if generator is not None:
            self.generator = generator
        else:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            self.generator = TextGenerator(
                model_loader=self.model_loader,
                config=config,
            )

    def validate_question(self, question: Any) -> str:
        """Validate input question string.

        Args:
            question: User input question.

        Returns:
            Normalized question string.

        Raises:
            InvalidQuestionError: If input is not a string, is empty, or exceeds max length.
        """
        if not isinstance(question, str):
            raise InvalidQuestionError(f"Question must be a string, got {type(question).__name__}.")

        cleaned = question.strip()
        if not cleaned:
            raise InvalidQuestionError("Question cannot be empty or whitespace-only.")

        if len(cleaned) > MAX_QUESTION_LENGTH:
            raise InvalidQuestionError(
                f"Question exceeds maximum allowed length of {MAX_QUESTION_LENGTH} characters "
                f"(received {len(cleaned)} characters)."
            )

        return cleaned

    def generate(self, question: str) -> AnswerResult:
        """Execute baseline question answering and return standardized AnswerResult.

        Args:
            question: Plain-text cybersecurity question.

        Returns:
            AnswerResult with pipeline='base', sources=[], and runtime metadata.
        """
        clean_question = self.validate_question(question)

        # Measure wall-clock generation latency
        t0 = time.perf_counter()
        gen_result = self.generator.generate(
            clean_question,
            prompt_version=self.prompt_version,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metadata: dict[str, Any] = {
            "prompt_version": self.prompt_version,
            "device": str(self.model_loader.device),
            "dtype": str(self.model_loader.dtype),
            "input_token_count": gen_result.input_token_count,
            "output_token_count": gen_result.output_token_count,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        }

        return AnswerResult(
            question=clean_question,
            pipeline="base",
            answer=gen_result.answer,
            model_id=self.model_loader.model_id,
            latency_ms=latency_ms,
            sources=[],  # Base pipeline receives no retrieval context
            metadata=metadata,
        )

    def __call__(self, question: str) -> AnswerResult:
        """Convenience callable alias for generate()."""
        return self.generate(question)
