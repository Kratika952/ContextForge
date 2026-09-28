"""
Base LLM Model Loader for ContextForge.
=======================================
Loads and caches the canonical Qwen/Qwen2.5-1.5B-Instruct causal language model
and tokenizer with explicit device and dtype strategy.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from contextforge.common.types import ModelLoadError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_device(device_pref: str = "auto") -> torch.device:
    """Resolve compute device based on preference and hardware availability.

    Args:
        device_pref: 'auto', 'cuda', 'cpu', or specific device string.

    Returns:
        torch.device instance.
    """
    pref = (device_pref or "auto").lower().strip()
    if pref in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device(pref)


def resolve_dtype(device: torch.device, dtype_pref: str = "auto") -> torch.dtype:
    """Resolve model weights data type based on target device and hardware capabilities.

    Args:
        device: Resolved torch.device.
        dtype_pref: 'auto', 'float32', 'bfloat16', 'float16', or torch.dtype.

    Returns:
        torch.dtype instance.
    """
    if isinstance(dtype_pref, torch.dtype):
        return dtype_pref

    pref = (dtype_pref or "auto").lower().strip()

    if pref in ("auto", ""):
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # On CPU, bfloat16 is supported in PyTorch 2.x and saves 50% memory
        return torch.bfloat16

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if pref in dtype_map:
        return dtype_map[pref]

    logger.warning("Unrecognized dtype preference '%s', falling back to bfloat16", pref)
    return torch.bfloat16


class BaseModelLoader:
    """Manages loading, device placement, and caching of the canonical Qwen2.5 base LLM."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize BaseModelLoader.

        Args:
            model_id: HuggingFace repository ID.
            device: 'auto', 'cuda', or 'cpu'.
            dtype: 'auto', 'float32', 'bfloat16', or 'float16'.
            trust_remote_code: HuggingFace security setting.
            model: Pre-loaded model instance (e.g. for testing).
            tokenizer: Pre-loaded tokenizer instance (e.g. for testing).
        """
        self.model_id = model_id
        self.device_pref = device
        self.dtype_pref = dtype
        self.trust_remote_code = trust_remote_code

        self._resolved_device = resolve_device(device)
        self._resolved_dtype = resolve_dtype(self._resolved_device, dtype)

        self._model = model
        self._tokenizer = tokenizer

        # If model was passed in, ensure it is in eval mode
        if self._model is not None and hasattr(self._model, "eval"):
            self._model.eval()

    @property
    def is_loaded(self) -> bool:
        """Return True if both tokenizer and model are loaded into memory."""
        return self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> torch.device:
        """Return active execution device."""
        return self._resolved_device

    @property
    def dtype(self) -> torch.dtype:
        """Return active model weight dtype."""
        return self._resolved_dtype

    @property
    def model(self) -> Any:
        """Return loaded model, loading on-demand if needed."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Return loaded tokenizer, loading on-demand if needed."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def load(self) -> None:
        """Load tokenizer and causal language model into memory.

        Raises:
            ModelLoadError: If download, initialization, or device placement fails.
        """
        if self.is_loaded:
            return

        try:
            logger.info("Loading tokenizer for '%s'...", self.model_id)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                )

            # Ensure tokenizer has padding token configured
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            logger.info(
                "Loading model '%s' on %s (%s)...",
                self.model_id,
                self._resolved_device,
                self._resolved_dtype,
            )
            if self._model is None:
                load_kwargs: dict[str, Any] = {
                    "trust_remote_code": self.trust_remote_code,
                }
                # Support Transformers v5+ dtype parameter or torch_dtype
                if (
                    hasattr(transformers, "__version__")
                    and int(transformers.__version__.split(".")[0]) >= 5
                ):
                    load_kwargs["dtype"] = self._resolved_dtype
                else:
                    load_kwargs["torch_dtype"] = self._resolved_dtype

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )

                # Move model to resolved device
                model = model.to(self._resolved_device)

                # Put model in eval mode
                model.eval()
                self._model = model

            logger.info("Model '%s' successfully loaded into memory.", self.model_id)

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model or tokenizer '{self.model_id}': {e}"
            ) from e

    def get_runtime_metadata(self) -> dict[str, Any]:
        """Return runtime diagnostic metadata for inference records."""
        param_count = None
        if self._model is not None and hasattr(self._model, "parameters"):
            param_count = sum(p.numel() for p in self._model.parameters())

        return {
            "model_id": self.model_id,
            "device": str(self._resolved_device),
            "dtype": str(self._resolved_dtype),
            "parameter_count": param_count,
            "is_loaded": self.is_loaded,
        }
