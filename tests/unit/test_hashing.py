"""Unit tests for artifact SHA-256 hashing."""

from pathlib import Path
from contextforge.ingestion.pdf_loader import compute_sha256


def test_sha256_identical_bytes(tmp_path: Path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"

    payload = b"Exact identical cybersecurity content for reproducibility testing."
    f1.write_bytes(payload)
    f2.write_bytes(payload)

    hash1 = compute_sha256(f1)
    hash2 = compute_sha256(f2)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_sha256_different_bytes(tmp_path: Path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"

    f1.write_bytes(b"Version 1.0 of defensive policy.")
    f2.write_bytes(b"Version 2.0 of defensive policy.")

    hash1 = compute_sha256(f1)
    hash2 = compute_sha256(f2)

    assert hash1 != hash2
