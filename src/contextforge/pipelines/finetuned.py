"""Fine-Tuned LLM Pipeline for ContextForge.
=========================================
Executes domain question answering using the canonical base LLM augmented
with a PEFT LoRA / QLoRA adapter trained on domain-specific QA examples.
Returns standardized AnswerResult without retrieval context.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from contextforge.common.types import AnswerResult, InvalidQuestionError, ModelLoadError
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.generation import GenerationConfig, TextGenerator

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = "artifacts/finetuning/phase8_pilot"
DEFAULT_PROMPT_VERSION = "base_v1"
MAX_QUESTION_LENGTH = 4000


class FineTunedPipeline:
    """Fine-tuned LLM inference pipeline using PEFT adapter augmentation."""

    def __init__(
        self,
        adapter_path: str | Path = DEFAULT_ADAPTER_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        model_loader: BaseModelLoader | None = None,
        generator: TextGenerator | None = None,
    ) -> None:
        """Initialize FineTunedPipeline.

        Args:
            adapter_path: Directory path containing the trained PEFT LoRA adapter.
            model_id: Base model repository ID.
            device: Compute device ('auto', 'cuda', 'cpu').
            dtype: Weight dtype ('auto', 'float32', 'bfloat16', 'float16').
            max_new_tokens: Maximum tokens to generate.
            prompt_version: System prompt template identifier.
            model_loader: Pre-configured BaseModelLoader instance (optional).
            generator: Pre-configured TextGenerator instance (optional).
        """
        self.adapter_path = Path(adapter_path)
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

        self._adapter_loaded = False
        self._load_adapter()

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

    def _load_adapter(self) -> None:
        """Attach PEFT LoRA adapter weights to the loaded base model."""
        if not self.adapter_path.exists():
            logger.warning(
                "Adapter path '%s' does not exist yet. Operating in unadapted base mode until trained.",
                self.adapter_path,
            )
            return

        try:
            from peft import PeftModel

            # Ensure base model is loaded
            base_model = self.model_loader.model
            logger.info("Loading PEFT adapter from '%s'...", self.adapter_path)
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
            )
            self.model_loader._model = peft_model
            self._adapter_loaded = True
            logger.info("PEFT adapter loaded successfully.")
        except Exception as e:
            logger.error("Failed to load adapter from '%s': %s", self.adapter_path, e)
            raise ModelLoadError(f"Failed to load PEFT adapter: {e}") from e

    @property
    def is_adapter_active(self) -> bool:
        """Return True if PEFT adapter weights are actively attached."""
        return self._adapter_loaded

    def validate_question(self, question: Any) -> str:
        """Validate input question string."""
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
        """Execute fine-tuned question answering and return standardized AnswerResult."""
        clean_question = self.validate_question(question)

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
            "adapter_path": str(self.adapter_path),
            "adapter_active": self._adapter_loaded,
            "input_token_count": gen_result.input_token_count,
            "output_token_count": gen_result.output_token_count,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        }

        return AnswerResult(
            question=clean_question,
            pipeline="finetuned",
            answer=gen_result.answer,
            model_id=f"{self.model_id}+lora",
            latency_ms=latency_ms,
            sources=[],  # Fine-tuned pipeline receives no retrieval context
            metadata=metadata,
        )

    def __call__(self, question: str) -> AnswerResult:
        """Convenience callable alias for generate()."""
        return self.generate(question)
