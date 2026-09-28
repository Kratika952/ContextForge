"""
FAISS Vector Index Store for ContextForge.
=========================================
Manages exact inner-product vector indexing (equivalent to cosine similarity
on normalized vectors) and metadata persistence for fast dense retrieval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from contextforge.common.types import (
    IncompatibleIndexError,
    IndexNotFoundError,
    RetrievalError,
)

INDEX_FILENAME = "index.faiss"
MAPPING_FILENAME = "mapping.json"
METADATA_FILENAME = "metadata.json"
EXPECTED_DIMENSION = 384


class FAISSStore:
    """Manages FAISS IndexFlatIP vector storage, persistence, and querying."""

    def __init__(
        self,
        dimension: int = EXPECTED_DIMENSION,
        model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_profile: str = "medium",
    ) -> None:
        """Initialize FAISSStore.

        Args:
            dimension: Embedding vector dimension (384 for bge-small-en-v1.5).
            model_name: Expected embedding model name.
            chunk_profile: Chunking profile associated with this index.
        """
        self.dimension = dimension
        self.model_name = model_name
        self.chunk_profile = chunk_profile

        self.index: faiss.IndexFlatIP | None = None
        self.mapping: list[str] = []  # vector_idx -> chunk_id
        self.metadata: dict[str, Any] = {}

    @property
    def is_built(self) -> bool:
        """Return True if an index is built and contains vectors."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in index."""
        return self.index.ntotal if self.index is not None else 0

    def build(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        chunk_profile: str | None = None,
    ) -> None:
        """Build an in-memory FAISS flat inner-product index from vectors.

        Args:
            embeddings: 2D numpy array of shape (N, dimension) with float32 dtype.
            chunk_ids: List of N chunk IDs corresponding 1-to-1 with embeddings rows.
            chunk_profile: Optional profile override.

        Raises:
            RetrievalError: If embeddings and chunk_ids shapes do not align.
        """
        if chunk_profile is not None:
            self.chunk_profile = chunk_profile

        if embeddings.ndim != 2:
            raise RetrievalError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        num_vectors, dim = embeddings.shape
        if dim != self.dimension:
            raise RetrievalError(
                f"Embedding dimension {dim} does not match expected store dimension {self.dimension}"
            )

        if num_vectors != len(chunk_ids):
            raise RetrievalError(
                f"Mismatch: received {num_vectors} vectors but {len(chunk_ids)} chunk IDs"
            )

        # Ensure contiguous float32 array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Build exact inner product index
        index = faiss.IndexFlatIP(self.dimension)
        if num_vectors > 0:
            index.add(vectors)

        self.index = index
        self.mapping = list(chunk_ids)
        self.metadata = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metric": "inner_product",
            "normalized": True,
            "chunk_profile": self.chunk_profile,
            "total_vectors": num_vectors,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, directory: Path | str) -> Path:
        """Persist index, ID mapping, and build metadata to directory.

        Args:
            directory: Path to output directory.

        Returns:
            Resolved Path to directory.

        Raises:
            RetrievalError: If index is not yet built.
        """
        if self.index is None:
            raise RetrievalError("Cannot save uninitialized index. Call build() first.")

        target_dir = Path(directory).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        index_path = target_dir / INDEX_FILENAME
        mapping_path = target_dir / MAPPING_FILENAME
        metadata_path = target_dir / METADATA_FILENAME

        # Write binary FAISS index
        faiss.write_index(self.index, str(index_path))

        # Write vector-to-chunk mapping
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=2)

        # Write metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        return target_dir

    def load(self, directory: Path | str, validate_profile: bool = True) -> None:
        """Load and validate persisted FAISS index, mapping, and metadata.

        Args:
            directory: Directory containing index.faiss, mapping.json, and metadata.json.
            validate_profile: Whether to enforce chunk_profile match.

        Raises:
            IndexNotFoundError: If directory or required files are missing.
            IncompatibleIndexError: If metadata attributes or vector counts mismatch.
        """
        target_dir = Path(directory).resolve()
        if not target_dir.is_dir():
            raise IndexNotFoundError(f"Index directory does not exist: {target_dir}")

        index_path = target_dir / INDEX_FILENAME
        mapping_path = target_dir / MAPPING_FILENAME
        metadata_path = target_dir / METADATA_FILENAME

        for file_path, name in [
            (index_path, INDEX_FILENAME),
            (mapping_path, MAPPING_FILENAME),
            (metadata_path, METADATA_FILENAME),
        ]:
            if not file_path.is_file():
                raise IndexNotFoundError(f"Missing required index artifact '{name}' in {target_dir}")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Validate metadata compatibility
        if meta.get("dimension") != self.dimension:
            raise IncompatibleIndexError(
                f"Index dimension mismatch: expected {self.dimension}, found {meta.get('dimension')}"
            )

        if meta.get("model_name") != self.model_name:
            raise IncompatibleIndexError(
                f"Embedding model mismatch: expected '{self.model_name}', found '{meta.get('model_name')}'"
            )

        if validate_profile and meta.get("chunk_profile") != self.chunk_profile:
            raise IncompatibleIndexError(
                f"Chunk profile mismatch: expected '{self.chunk_profile}', found '{meta.get('chunk_profile')}'"
            )

        # Load ID mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # Read FAISS index
        index = faiss.read_index(str(index_path))

        # Validate count consistency
        if index.ntotal != len(mapping):
            raise IncompatibleIndexError(
                f"Index count ({index.ntotal}) does not match mapping count ({len(mapping)})"
            )

        self.index = index
        self.mapping = mapping
        self.metadata = meta
        self.chunk_profile = meta.get("chunk_profile", self.chunk_profile)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, int]]:
        """Search the FAISS index with a query vector.

        Args:
            query_vector: 2D numpy array of shape (1, dimension) or 1D (dimension,).
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (chunk_id, similarity_score, rank_1_indexed) tuples.

        Raises:
            RetrievalError: If index is uninitialized or vector shape is invalid.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.shape[1] != self.dimension:
            raise RetrievalError(
                f"Query vector dimension {query_vector.shape[1]} does not match index dimension {self.dimension}"
            )

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(q_vec, k)

        results: list[tuple[str, float, int]] = []
        for rank_idx, (vec_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if vec_idx < 0 or vec_idx >= len(self.mapping):
                continue
            chunk_id = self.mapping[vec_idx]
            results.append((chunk_id, float(score), rank_idx))

        return results
