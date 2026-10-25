"""Pipeline manager and lifecycle controller for ContextForge.
=============================================================
Ensures predictable, lazy model loading and safe GPU VRAM switching
between Base, RAG, and Fine-Tuned pipelines (Sections 4 & 5).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Literal
from pathlib import Path

from contextforge.common.types import AnswerResult
from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.rag import RagPipeline
from contextforge.pipelines.finetuned import FineTunedPipeline

logger = logging.getLogger(__name__)

PipelineType = BasePipeline | RagPipeline | FineTunedPipeline
PipelineName = Literal["base", "rag", "finetuned"]


class PipelineManager:
    """Manages lazy initialization, caching, and safe GPU unloading across inference pipelines."""

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "auto",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str | Path = "artifacts/finetuning/phase8_pilot",
        top_k: int = 5,
        max_context_tokens: int = 1500,
        chunk_profile: str = "medium",
        db_path: Path | str | None = None,
        index_dir: Path | str | None = None,
    ) -> None:
        """Initialize PipelineManager.

        Args:
            device: Target compute device ('auto', 'cuda', 'cpu').
            dtype: Model weight dtype ('auto', 'bfloat16', 'float32').
            model_id: HuggingFace model repository identifier.
            adapter_path: Path to trained PEFT LoRA adapter directory.
            top_k: Number of retrieved chunks for RAG.
            max_context_tokens: Context token budget for RAG.
            chunk_profile: Chunk profile for RAG retrieval.
            db_path: Path to SQLite database.
            index_dir: Path to FAISS vector index directory.
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.adapter_path = Path(adapter_path)
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.chunk_profile = chunk_profile
        self.db_path = db_path
        self.index_dir = index_dir

        self._current_name: PipelineName | None = None
        self._current_instance: Any = None
        self._factories: dict[str, Callable[[], Any]] = {}

    @property
    def active_pipeline_name(self) -> str | None:
        """Return the name of the currently resident pipeline, if any."""
        return self._current_name

    def set_custom_pipeline(self, name: PipelineName, pipeline_instance: Any) -> None:
        """Inject a pre-built or mocked pipeline for testing."""
        self._current_name = name
        self._current_instance = pipeline_instance

    def set_factory(self, name: PipelineName, factory: Callable[[], Any]) -> None:
        """Register a custom factory for creating a specific pipeline (e.g. for testing)."""
        self._factories[name] = factory

    def unload_current(self) -> None:
        """Unload resident pipeline and reclaim GPU memory."""
        if self._current_instance is not None:
            logger.info("Unloading resident pipeline: %s", self._current_name)
            self._current_instance = None
            self._current_name = None

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

    def get_pipeline(self, name: PipelineName) -> Any:
        """Obtain the requested inference pipeline, reusing current instance if matching.

        Args:
            name: Target pipeline identifier ('base', 'rag', or 'finetuned').

        Returns:
            Callable pipeline instance exposing generate(question).

        Raises:
            ValueError: If an unrecognized pipeline name is requested.
            ModelLoadError / Exception: If model initialization fails.
        """
        if name not in ("base", "rag", "finetuned"):
            raise ValueError(f"Invalid pipeline '{name}'. Supported pipelines: 'base', 'rag', 'finetuned'.")

        # Reuse currently loaded pipeline if already active
        if self._current_name == name and self._current_instance is not None:
            return self._current_instance

        # Unload previous pipeline to avoid GPU memory accumulation
        self.unload_current()

        logger.info("Initializing requested pipeline: %s", name)

        # Check if custom factory was registered (e.g. for unit tests)
        if name in self._factories:
            instance = self._factories[name]()
            self._current_instance = instance
            self._current_name = name
            return instance

        # Default real pipeline creation
        if name == "base":
            instance = BasePipeline(
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
            )
        elif name == "rag":
            instance = RagPipeline(
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
                top_k=self.top_k,
                max_context_tokens=self.max_context_tokens,
                chunk_profile=self.chunk_profile,
                db_path=self.db_path,
                index_dir=self.index_dir,
            )
        elif name == "finetuned":
            instance = FineTunedPipeline(
                model_id=self.model_id,
                adapter_path=self.adapter_path,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unsupported pipeline: {name}")

        self._current_instance = instance
        self._current_name = name
        return instance
