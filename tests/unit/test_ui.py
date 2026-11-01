"""Unit tests for ContextForge Streamlit Frontend Client and UI logic.
===================================================================
Phase 11: Validates decoupled HTTP client, error handling,
frozen Phase 9 metrics display, and architecture boundary enforcement.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from contextforge.ui.app import (
    FROZEN_PHASE9_METRICS,
    PIPELINE_OPTIONS,
    collapse_display_sources,
    extract_response_metrics,
    extract_source_metadata,
)
from contextforge.ui.client import (
    ApiConnectionError,
    ApiError,
    ApiResponseError,
    ContextForgeClient,
    DEFAULT_API_URL,
)


class TestContextForgeClient:
    """Test suite for ContextForgeClient HTTP communication."""

    def test_client_init_defaults(self, monkeypatch):
        """Client should default to DEFAULT_API_URL when env is unset."""
        monkeypatch.delenv("CONTEXTFORGE_API_URL", raising=False)
        client = ContextForgeClient()
        assert client.base_url == DEFAULT_API_URL.rstrip("/")
        assert client.timeout == 180.0

    def test_client_init_custom_and_env(self, monkeypatch):
        """Client should prioritize explicit base_url, then CONTEXTFORGE_API_URL."""
        monkeypatch.setenv("CONTEXTFORGE_API_URL", "http://env-host:9000/")
        client_env = ContextForgeClient()
        assert client_env.base_url == "http://env-host:9000"

        client_custom = ContextForgeClient(base_url="http://custom-host:8080/")
        assert client_custom.base_url == "http://custom-host:8080"

    def test_check_health_success(self):
        """check_health returns deserialized JSON on HTTP 200."""
        mock_response = io.BytesIO(json.dumps({"status": "ok", "version": "0.1.0"}).encode("utf-8"))
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_cm):
            client = ContextForgeClient("http://127.0.0.1:8000")
            result = client.check_health()
            assert result == {"status": "ok", "version": "0.1.0"}

    def test_check_health_connection_error(self):
        """check_health raises ApiConnectionError on URLError."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            client = ContextForgeClient("http://127.0.0.1:8000")
            with pytest.raises(ApiConnectionError) as exc_info:
                client.check_health()
            assert "Could not connect to ContextForge API" in str(exc_info.value)

    def test_check_health_http_error(self):
        """check_health raises ApiResponseError on HTTP 500."""
        fp = io.BytesIO(b"Internal Server Error")
        http_err = urllib.error.HTTPError("http://127.0.0.1:8000/health", 500, "Server Error", {}, fp)
        with patch("urllib.request.urlopen", side_effect=http_err):
            client = ContextForgeClient("http://127.0.0.1:8000")
            with pytest.raises(ApiResponseError) as exc_info:
                client.check_health()
            assert exc_info.value.status_code == 500

    def test_ask_validation_empty_question(self):
        """ask raises ValueError on empty or whitespace question."""
        client = ContextForgeClient()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            client.ask("", "base")
        with pytest.raises(ValueError, match="Question cannot be empty"):
            client.ask("   \n\t ", "rag")

    def test_ask_validation_invalid_pipeline(self):
        """ask raises ValueError on unrecognized pipeline name."""
        client = ContextForgeClient()
        with pytest.raises(ValueError, match="Invalid pipeline 'unknown'"):
            client.ask("Valid question?", "unknown")

    def test_ask_success_base_pipeline(self):
        """ask sends POST request and returns expected payload."""
        expected = {
            "question": "What is OWASP?",
            "answer": "OWASP is an open web security community.",
            "pipeline": "base",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "latency_ms": 1250.5,
            "sources": [],
            "token_usage": {"input_tokens": 15, "output_tokens": 30},
        }
        mock_response = io.BytesIO(json.dumps(expected).encode("utf-8"))
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            client = ContextForgeClient("http://127.0.0.1:8000")
            res = client.ask("What is OWASP?", "base")
            assert res == expected

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8000/ask"
            assert req.get_method() == "POST"
            assert req.headers["Content-type"] == "application/json"

    def test_ask_connection_error(self):
        """ask raises ApiConnectionError when server is offline."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            client = ContextForgeClient("http://127.0.0.1:8000")
            with pytest.raises(ApiConnectionError) as exc_info:
                client.ask("What is CSF?", "rag")
            assert "Connection failed to" in str(exc_info.value)

    def test_ask_http_error_with_json_detail(self):
        """ask unpacks structured error detail from backend HTTPError."""
        detail_json = json.dumps({"detail": "Model failed to load on GPU"}).encode("utf-8")
        fp = io.BytesIO(detail_json)
        http_err = urllib.error.HTTPError("http://127.0.0.1:8000/ask", 500, "Error", {}, fp)

        with patch("urllib.request.urlopen", side_effect=http_err):
            client = ContextForgeClient("http://127.0.0.1:8000")
            with pytest.raises(ApiResponseError) as exc_info:
                client.ask("What is CSF?", "finetuned")
            assert exc_info.value.status_code == 500
            assert "Model failed to load on GPU" in exc_info.value.detail


class TestArchitectureIsolation:
    """Verify frontend code does not import heavy ML/inference libraries."""

    def test_ui_client_has_no_ml_imports(self):
        """client.py must only use standard library networking."""
        import contextforge.ui.client as client_module
        module_text = open(client_module.__file__, "r", encoding="utf-8").read()

        prohibited = [
            "torch",
            "transformers",
            "sentence_transformers",
            "faiss",
            "peft",
            "trl",
            "BasePipeline",
            "RagPipeline",
            "FineTunedPipeline",
            "cuda",
        ]
        for term in prohibited:
            assert f"import {term}" not in module_text
            assert f"from {term}" not in module_text

    def test_ui_app_has_no_ml_imports(self):
        """app.py must only use streamlit and client.py."""
        import contextforge.ui.app as app_module
        module_text = open(app_module.__file__, "r", encoding="utf-8").read()

        prohibited = [
            "torch",
            "transformers",
            "sentence_transformers",
            "faiss",
            "peft",
            "trl",
            "BasePipeline",
            "RagPipeline",
            "FineTunedPipeline",
        ]
        for term in prohibited:
            assert f"import {term}" not in module_text
            assert f"from {term}" not in module_text


class TestFrozenPhase9Metrics:
    """Verify frozen Phase 9 metrics match verified experiment values."""

    def test_frozen_metrics_values(self):
        """Values in FROZEN_PHASE9_METRICS must match Phase 9 report exactly."""
        p_data = FROZEN_PHASE9_METRICS["pipelines"]

        # Base LLM
        assert p_data["Base LLM"]["token_f1"] == 0.2870
        assert p_data["Base LLM"]["semantic_similarity"] == 0.8641
        assert p_data["Base LLM"]["mean_latency_ms"] == 8784.4

        # RAG Pipeline
        assert p_data["RAG Pipeline"]["token_f1"] == 0.3159
        assert p_data["RAG Pipeline"]["semantic_similarity"] == 0.8715
        assert p_data["RAG Pipeline"]["mean_latency_ms"] == 11837.3
        assert p_data["RAG Pipeline"]["hit_at_5"] == 1.0000
        assert p_data["RAG Pipeline"]["mrr"] == 0.7722
        assert p_data["RAG Pipeline"]["citation_presence"] == 0.3333
        assert p_data["RAG Pipeline"]["citation_validity"] == 1.0000

        # Fine-Tuned LLM
        assert p_data["Fine-Tuned LLM"]["token_f1"] == 0.3274
        assert p_data["Fine-Tuned LLM"]["semantic_similarity"] == 0.8692
        assert p_data["Fine-Tuned LLM"]["mean_latency_ms"] == 5633.3

    def test_pipeline_options_keys(self):
        """Pipeline options must map to base, rag, finetuned."""
        assert PIPELINE_OPTIONS == {
            "Base LLM": "base",
            "RAG Pipeline": "rag",
            "Fine-Tuned LLM": "finetuned",
        }


class TestSourceMetadataExtraction:
    """Validate extraction of real metadata without fake 'Unknown' or 'N/A' placeholders."""

    def test_extract_real_document_id_and_title(self):
        """extract_source_metadata resolves real document_id and title from metadata."""
        src = {
            "chunk_id": "nist-csf-20_medium_0062",
            "document_id": "nist-csf-20",
            "text": "The Govern function informs risk management.",
            "score": 0.8842,
            "rank": 1,
            "source_label": "[nist-csf-20-P27]",
            "metadata": {
                "document_title": "The NIST Cybersecurity Framework (CSF) 2.0",
                "page_start": 27,
                "page_end": 27,
                "section_heading": None,
            },
        }
        info = extract_source_metadata(src)
        assert info["doc_id"] == "nist-csf-20"
        assert info["title"] == "The NIST Cybersecurity Framework (CSF) 2.0"
        assert info["source_label"] == "[nist-csf-20-P27]"
        assert info["score"] == 0.8842
        assert info["excerpt"] == "The Govern function informs risk management."
        assert info["page"] == "27"
        assert info["section"] is None  # no fake "N/A"

    def test_extract_page_range(self):
        """Multi-page chunks format as page ranges."""
        src = {
            "chunk_id": "cisa_01",
            "document_id": "cisa-cpg-20",
            "text": "CPG guidelines excerpt.",
            "metadata": {"page_start": 7, "page_end": 8},
        }
        info = extract_source_metadata(src)
        assert info["page"] == "7–8"

    def test_extract_section_when_present(self):
        """Real section heading is extracted; literal 'N/A' or 'None' is omitted."""
        src_with_sec = {
            "document_id": "owasp-top-10",
            "text": "A01: Broken Access Control",
            "metadata": {"section_heading": "A01:2025 - Broken Access Control"},
        }
        info = extract_source_metadata(src_with_sec)
        assert info["section"] == "A01:2025 - Broken Access Control"

        src_na_sec = {
            "document_id": "owasp-top-10",
            "text": "A01 text",
            "metadata": {"section_heading": "N/A"},
        }
        assert extract_source_metadata(src_na_sec)["section"] is None

    def test_fallback_when_title_absent(self):
        """Falls back to document_id if document_title is missing."""
        src = {
            "document_id": "nist-csf-20",
            "text": "Excerpt",
            "metadata": {},
        }
        info = extract_source_metadata(src)
        assert info["title"] == "nist-csf-20"
        assert info["doc_id"] == "nist-csf-20"


class TestDuplicateSourceCollapsing:
    """Validate presentation-only duplicate collapsing for identical excerpts."""

    def test_collapses_identical_excerpts_preserving_count(self):
        """Identical cover pages or repetitive excerpts collapse into 1 card."""
        raw_sources = [
            {
                "document_id": "nist-csf-20",
                "text": "Cover page boilerplate for NIST CSF 2.0",
                "score": 0.782,
                "source_label": "[nist-csf-20-P22]",
            },
            {
                "document_id": "nist-csf-20",
                "text": "Cover page boilerplate for NIST CSF 2.0",
                "score": 0.782,
                "source_label": "[nist-csf-20-P24]",
            },
            {
                "document_id": "nist-csf-20",
                "text": "Cover page boilerplate for NIST CSF 2.0",
                "score": 0.782,
                "source_label": "[nist-csf-20-P25]",
            },
            {
                "document_id": "nist-csf-20",
                "text": "Unique substantive text about Govern function outcomes.",
                "score": 0.810,
                "source_label": "[nist-csf-20-P10]",
            },
        ]

        collapsed = collapse_display_sources(raw_sources)
        assert len(collapsed) == 2  # 4 raw chunks -> 2 unique display cards
        assert collapsed[0]["_collapse_count"] == 3
        assert collapsed[0]["_labels"] == [
            "[nist-csf-20-P22]",
            "[nist-csf-20-P24]",
            "[nist-csf-20-P25]",
        ]
        assert collapsed[1]["_collapse_count"] == 1
        assert len(raw_sources) == 4  # underlying list untouched

    def test_distinct_sources_not_collapsed(self):
        """Distinct text chunks remain separate."""
        raw_sources = [
            {"document_id": "doc1", "text": "Excerpt A", "score": 0.9},
            {"document_id": "doc1", "text": "Excerpt B", "score": 0.8},
            {"document_id": "doc2", "text": "Excerpt A", "score": 0.7},  # different doc
        ]
        collapsed = collapse_display_sources(raw_sources)
        assert len(collapsed) == 3


class TestResponseMetricsExtraction:
    """Validate extraction of latency, tokens, and caption metadata."""

    def test_extracts_real_tokens_and_prompt_version(self):
        """Real input/output token counts and prompt version are extracted from metadata."""
        resp = {
            "latency_ms": 13385.4,
            "pipeline": "rag",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "metadata": {
                "prompt_version": "rag_v1",
                "input_token_count": 605,
                "output_token_count": 83,
            },
        }
        metrics, captions = extract_response_metrics(resp, "rag")
        metric_dict = dict(metrics)
        assert metric_dict["Latency"] == "13385.4 ms"
        assert metric_dict["Pipeline"] == "rag"
        assert metric_dict["Input Tokens"] == "605"
        assert metric_dict["Output Tokens"] == "83"
        assert "Model ID: `Qwen/Qwen2.5-1.5B-Instruct`" in captions
        assert "Prompt Version: `rag_v1`" in captions

    def test_omits_missing_tokens_no_fake_na(self):
        """If token counts are absent, they are omitted instead of displaying 'N/A'."""
        resp = {
            "latency_ms": 250.0,
            "pipeline": "base",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "metadata": {},
        }
        metrics, captions = extract_response_metrics(resp, "base")
        metric_labels = [m[0] for m in metrics]
        assert "Input Tokens" not in metric_labels
        assert "Output Tokens" not in metric_labels
        assert len(metrics) == 2  # Latency and Pipeline only

