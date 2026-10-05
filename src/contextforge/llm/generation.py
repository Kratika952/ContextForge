"""
Text Generation Wrapper for ContextForge.
=========================================
Coordinates prompt template formatting, tokenization, model generation,
and isolated response token decoding with deterministic generation controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from contextforge.common.types import GenerationError
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.prompts import format_base_messages

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 256


@dataclass
class GenerationConfig:
    """Generation parameters for deterministic baseline inference."""

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None

    def to_generate_kwargs(self, pad_token_id: int | None = None, eos_token_id: int | None = None) -> dict[str, Any]:
        """Convert to kwargs dict for model.generate(), omitting invalid sampling args."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id

        # Only pass sampling parameters when do_sample is True
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p

        return kwargs


@dataclass
class GenerationResult:
    """Output contract for model text generation."""

    answer: str
    input_token_count: int
    output_token_count: int
    raw_prompt: str


class TextGenerator:
    """Generates answers using BaseModelLoader and official Qwen chat templates."""

    def __init__(
        self,
        model_loader: BaseModelLoader,
        config: GenerationConfig | None = None,
    ) -> None:
        """Initialize TextGenerator.

        Args:
            model_loader: Loaded or loadable BaseModelLoader.
            config: GenerationConfig instance (defaults to deterministic do_sample=False).
        """
        self.model_loader = model_loader
        self.config = config or GenerationConfig()

    def generate(
        self,
        question: str | None = None,
        prompt_version: str = "base_v1",
        messages: list[dict[str, str]] | None = None,
    ) -> GenerationResult:
        """Generate response for a cybersecurity question or pre-formatted message list.

        Args:
            question: Cleaned user query (optional if messages is provided).
            prompt_version: System prompt version (used if formatting from question).
            messages: Pre-formatted chat message dictionaries (e.g. from RAG formatter).

        Returns:
            GenerationResult containing decoded answer and token counts.

        Raises:
            GenerationError: If generation execution fails or produces empty output.
        """
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        device = self.model_loader.device

        # 1. Resolve messages
        if messages is None:
            if not question:
                raise GenerationError("Either 'question' or 'messages' must be provided for generation.")
            messages = format_base_messages(question, prompt_version=prompt_version)

        # 2. Apply official Qwen chat template

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            raise GenerationError(f"Failed to apply chat template: {e}") from e

        # 3. Tokenize input prompt
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            input_token_count = input_ids.shape[1]
        except Exception as e:
            raise GenerationError(f"Tokenization failed: {e}") from e

        # 4. Prepare generation kwargs
        gen_kwargs = self.config.to_generate_kwargs(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 5. Execute model.generate
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
        except Exception as e:
            raise GenerationError(f"Model generation forward pass failed: {e}") from e

        # 6. Isolate newly generated tokens (strip input prompt tokens)
        try:
            output_tokens = outputs[0][input_token_count:]
            output_token_count = len(output_tokens)
            answer = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            raise GenerationError(f"Failed to decode output tokens: {e}") from e

        if not answer:
            raise GenerationError("Model returned empty generated text.")

        return GenerationResult(
            answer=answer,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            raw_prompt=prompt_text,
        )
