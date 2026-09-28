"""Unit tests for the conservative text cleaner."""

import pytest
from contextforge.ingestion.cleaner import clean_text


def test_cleaner_normalizes_whitespace_and_excessive_newlines():
    raw = "Paragraph 1 with   multiple    spaces.\n\n\n\n\nParagraph 2."
    cleaned = clean_text(raw)
    assert cleaned == "Paragraph 1 with multiple spaces.\n\nParagraph 2."


def test_cleaner_preserves_cybersecurity_tokens():
    cyber_text = (
        "Vulnerability CVE-2025-12345 detected.\n"
        "Protocol: TLS 1.3 / HTTP/2 over port 443.\n"
        "Vector: <script>alert('XSS')</script> in /api/v1/auth?token=abc_xyz:999.\n"
        "Query: SELECT * FROM users WHERE role = 'admin' --"
    )
    cleaned = clean_text(cyber_text)
    # Ensure all sensitive cybersecurity symbols, casing, and punctuation survived
    assert "CVE-2025-12345" in cleaned
    assert "TLS 1.3 / HTTP/2" in cleaned
    assert "<script>alert('XSS')</script>" in cleaned
    assert "/api/v1/auth?token=abc_xyz:999" in cleaned
    assert "SELECT * FROM users WHERE role = 'admin' --" in cleaned


def test_cleaner_strips_control_characters():
    raw_with_control = "Text\x00with\x07bell\x1bescape."
    cleaned = clean_text(raw_with_control)
    assert cleaned == "Textwithbellescape."


def test_cleaner_normalizes_ligatures_via_nfkc():
    # \ufb01 is the 'fi' ligature, \ufb02 is 'fl'
    ligature_text = "Speci\ufb01c \ufb02ow"
    cleaned = clean_text(ligature_text)
    assert cleaned == "Specific flow"


def test_cleaner_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
