"""
Dense Embedding Wrapper for ContextForge.
=========================================
Wraps Sentence Transformers with the approved BAAI/bge-small-en-v1.5 model.
Enforces vector normalization (dimension 384) and handles the official BGE query prefix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from contextforge.common.types import EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EXPECTED_DIMENSION = 384


class BGEEmbedder:
    """Embedding generator wrapping BAAI/bge-small-en-v1.5 with SentenceTransformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize_embeddings: bool = True,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        device: str | None = None,
    ) -> None:
        """Initialize BGEEmbedder.

        Args:
            model_name: HuggingFace model identifier.
            normalize_embeddings: Whether to L2-normalize vectors (unit norm for cosine similarity).
            query_instruction: Official BGE retrieval prefix applied strictly to queries.
            device: Compute device ('cpu', 'cuda', etc.). If None, automatically determined.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction
        self.dimension = EXPECTED_DIMENSION

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}") from e

        # Validate embedding dimension
        if hasattr(self.model, "get_embedding_dimension"):
            test_dim = self.model.get_embedding_dimension()
        else:
            test_dim = self.model.get_sentence_embedding_dimension()
        if test_dim != self.dimension:
            logger.warning(
                "Embedding model dimension %d does not match expected %d",
                test_dim,
                self.dimension,
            )
            self.dimension = test_dim

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Compute dense embeddings for a batch of document/chunk passages.

        IMPORTANT: Passages do NOT receive the query instruction prefix.

        Args:
            texts: List of passage strings to embed.
            batch_size: Inference batch size.
            show_progress_bar: Whether to display a tqdm progress bar.

        Returns:
            2D numpy array of shape (N, dimension) with float32 dtype.

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Error computing passage embeddings: {e}") from e

    def embed_query(
        self,
        query: str,
        add_instruction: bool = True,
    ) -> np.ndarray:
        """Compute dense embedding for a single search query.

        Args:
            query: User search query text.
            add_instruction: Whether to prepend the official BGE query instruction.

        Returns:
            2D numpy array of shape (1, dimension) with float32 dtype.

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        query_text = query.strip()
        if add_instruction and self.query_instruction:
            query_text = f"{self.query_instruction}{query_text}"

        try:
            embedding = self.model.encode(
                query_text,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            # Ensure shape is (1, dimension)
            if embedding.ndim == 1:
                embedding = np.expand_dims(embedding, axis=0)
            return embedding.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Error computing query embedding: {e}") from e
