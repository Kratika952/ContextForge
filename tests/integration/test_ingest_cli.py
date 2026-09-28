"""Integration tests for scripts/ingest.py CLI."""

import subprocess
import sys
from pathlib import Path
import pymupdf


def create_sample_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(pymupdf.Point(50, 50), text)
    doc.save(str(path))
    doc.close()
    return path


def test_cli_ingests_pdf(tmp_path: Path):
    db_file = tmp_path / "cli_test.db"
    pdf_file = create_sample_pdf(tmp_path / "test_doc.pdf", "NIST CSF CLI Test Content.")

    cmd = [
        sys.executable,
        "scripts/ingest.py",
        str(pdf_file),
        "--db",
        str(db_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    stdout = res.stdout
    assert "Status:         INGESTED" in stdout
    assert "Source type:    PDF" in stdout
    assert "SHA-256:" in stdout
    assert "Database:" in stdout

    # Repeat run: should show ALREADY_INGESTED and exit 0
    res_dup = subprocess.run(cmd, capture_output=True, text=True)
    assert res_dup.returncode == 0
    assert "Status:         ALREADY_INGESTED" in res_dup.stdout


def test_cli_fails_on_missing_file(tmp_path: Path):
    db_file = tmp_path / "cli_test.db"
    cmd = [
        sys.executable,
        "scripts/ingest.py",
        str(tmp_path / "non_existent.pdf"),
        "--db",
        str(db_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Status: FAILED" in res.stdout
