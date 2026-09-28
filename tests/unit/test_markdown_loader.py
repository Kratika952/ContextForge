"""Unit tests for Markdown extraction module."""

from pathlib import Path
import pytest

from contextforge.common.types import (
    DocumentNotFoundError,
    NoExtractableTextError,
)
from contextforge.ingestion.markdown_loader import MarkdownLoader


def test_markdown_loader_loads_and_splits_sections(tmp_path: Path):
    md_path = tmp_path / "A01_Broken_Access.md"
    content = (
        "# Broken Access Control\n\n"
        "Access control enforces policy such that users cannot act outside intended permissions.\n\n"
        "## How to Prevent\n\n"
        "Deny by default and enforce record ownership.\n\n"
        "## Example Scenarios\n\n"
        "Attacker modifies URL parameter to access another account.\n"
    )
    md_path.write_text(content, encoding="utf-8")

    loader = MarkdownLoader()
    doc = loader.load(md_path)

    assert doc.source_type == "markdown"
    assert doc.title == "Broken Access Control"
    assert doc.total_units == 3
    assert len(doc.checksum) == 64
    assert doc.units[0].heading == "# Broken Access Control"
    assert "Access control enforces policy" in doc.units[0].text
    assert doc.units[1].heading == "## How to Prevent"
    assert "Deny by default" in doc.units[1].text
    assert doc.units[2].heading == "## Example Scenarios"


def test_markdown_loader_preserves_utf8_and_code(tmp_path: Path):
    md_path = tmp_path / "injection.md"
    content = (
        "# SQL Injection\n\n"
        "Vulnerability description with UTF-8: 🛡️ Security Check.\n\n"
        "```sql\n"
        "SELECT * FROM accounts WHERE id = 'user_input';\n"
        "```\n"
    )
    md_path.write_text(content, encoding="utf-8")

    loader = MarkdownLoader()
    doc = loader.load(md_path)

    assert "🛡️ Security Check" in doc.full_text
    assert "SELECT * FROM accounts WHERE id = 'user_input';" in doc.full_text


def test_markdown_loader_empty_raises_no_text(tmp_path: Path):
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("   \n\n  ", encoding="utf-8")

    loader = MarkdownLoader()
    with pytest.raises(NoExtractableTextError):
        loader.load(empty_md)


def test_markdown_loader_missing_raises_not_found(tmp_path: Path):
    loader = MarkdownLoader()
    with pytest.raises(DocumentNotFoundError):
        loader.load(tmp_path / "missing.md")
