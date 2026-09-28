"""
Unit tests for ContextForge FAISSStore.
=======================================
Verifies IndexFlatIP vector index build, persistence, query search,
and compatibility validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextforge.common.types import (
    IncompatibleIndexError,
    IndexNotFoundError,
    RetrievalError,
)
from contextforge.retrieval.faiss_store import FAISSStore


@pytest.fixture
def sample_vectors():
    """Generate 10 normalized random 384-dim vectors."""
    np.random.seed(42)
    vecs = np.random.randn(10, 384).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def chunk_ids():
    return [f"chunk_{i:03d}" for i in range(10)]


def test_faiss_store_build_and_search(sample_vectors, chunk_ids):
    store = FAISSStore(dimension=384, chunk_profile="medium")
    store.build(embeddings=sample_vectors, chunk_ids=chunk_ids, chunk_profile="medium")

    assert store.is_built is True
    assert store.total_vectors == 10

    # Search with the first vector (should match chunk_000 with score ~1.0)
    query_vec = sample_vectors[0:1]
    results = store.search(query_vec, top_k=3)

    assert len(results) == 3
    top_chunk, top_score, top_rank = results[0]
    assert top_chunk == "chunk_000"
    assert top_rank == 1
    assert pytest.approx(top_score, abs=1e-5) == 1.0


def test_faiss_store_save_and_load(tmp_path, sample_vectors, chunk_ids):
    store = FAISSStore(dimension=384, chunk_profile="medium")
    store.build(embeddings=sample_vectors, chunk_ids=chunk_ids)

    saved_dir = store.save(tmp_path / "index_test")
    assert (saved_dir / "index.faiss").is_file()
    assert (saved_dir / "mapping.json").is_file()
    assert (saved_dir / "metadata.json").is_file()

    # Load in a new store instance
    loaded_store = FAISSStore(dimension=384, chunk_profile="medium")
    loaded_store.load(saved_dir)

    assert loaded_store.is_built is True
    assert loaded_store.total_vectors == 10
    assert loaded_store.mapping == chunk_ids

    # Search loaded index
    results = loaded_store.search(sample_vectors[1:2], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "chunk_001"


def test_faiss_store_load_missing_directory_raises():
    store = FAISSStore(dimension=384)
    with pytest.raises(IndexNotFoundError):
        store.load("nonexistent_faiss_dir_98765")


def test_faiss_store_incompatible_dimension_raises(tmp_path, sample_vectors, chunk_ids):
    store = FAISSStore(dimension=384, chunk_profile="medium")
    store.build(embeddings=sample_vectors, chunk_ids=chunk_ids)
    saved_dir = store.save(tmp_path / "dim_test")

    # Try loading into a store expecting 768 dimensions
    diff_store = FAISSStore(dimension=768, chunk_profile="medium")
    with pytest.raises(IncompatibleIndexError, match="dimension mismatch"):
        diff_store.load(saved_dir)


def test_faiss_store_dimension_mismatch_during_build():
    store = FAISSStore(dimension=384)
    wrong_vecs = np.random.randn(5, 128).astype(np.float32)
    with pytest.raises(RetrievalError, match="dimension 128 does not match"):
        store.build(embeddings=wrong_vecs, chunk_ids=["c1", "c2", "c3", "c4", "c5"])
