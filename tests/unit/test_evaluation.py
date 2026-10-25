"""
Unit tests for ContextForge Phase 6 Evaluation Harness.
======================================================
Tests dataset parsing, metric calculations, expected-source matching,
refusal detection, aggregation, and JSON serialization.
Zero model loading (fast execution).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextforge.common.types import AnswerResult, RetrievedChunk
from contextforge.evaluation.evaluator import (
    EvaluationQuestion,
    aggregate_evaluation_results,
    calculate_reference_coverage,
    evaluate_question_pair,
    is_expected_source_retrieved,
    is_unsupported_refusal,
    load_evaluation_dataset,
)


def test_load_evaluation_dataset_valid(tmp_path: Path):
    """Test loading valid evaluation dataset from jsonl file."""
    data = [
        {"id": "q1", "question": "What is MFA?", "expected_source": "cisa", "reference_points": ["MFA", "security"]},
        {"id": "q2", "question": "What is cake?", "expected_source": "none", "reference_points": ["sugar"]},
    ]
    file_path = tmp_path / "test_eval.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    questions = load_evaluation_dataset(file_path)
    assert len(questions) == 2
    assert questions[0].id == "q1"
    assert questions[0].question == "What is MFA?"
    assert questions[0].expected_source == "cisa"
    assert questions[0].reference_points == ["MFA", "security"]


def test_load_evaluation_dataset_missing_fields_raises(tmp_path: Path):
    """Test that missing required fields raises ValueError."""
    file_path = tmp_path / "bad_eval.jsonl"
    file_path.write_text('{"id": "q1", "question": "No source"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected_source"):
        load_evaluation_dataset(file_path)


def test_load_evaluation_dataset_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_evaluation_dataset("non_existent_file_path.jsonl")


def test_calculate_reference_coverage():
    """Test rule-based reference point coverage calculation."""
    refs = ["Broken Access Control", "Security Misconfiguration", "Injection"]

    # Full match (case-insensitive)
    ans1 = "The risks are broken access control, security misconfiguration, and injection."
    assert calculate_reference_coverage(ans1, refs) == 1.0

    # Partial match (2/3)
    ans2 = "Broken access control and injection are common risks."
    assert calculate_reference_coverage(ans2, refs) == round(2 / 3, 4)

    # Zero match
    ans3 = "Nothing related to the items above."
    assert calculate_reference_coverage(ans3, refs) == 0.0

    # Empty reference list defaults to 1.0
    assert calculate_reference_coverage(ans1, []) == 1.0

    # Empty answer
    assert calculate_reference_coverage("", refs) == 0.0


def test_is_expected_source_retrieved():
    """Test source correctness matching against retrieved chunks."""
    chunk_owasp = RetrievedChunk(
        chunk_id="c1",
        document_id="a01-2025-broken-access-control",
        text="Sample text",
        score=0.85,
        rank=1,
        source_label="[SOURCE_1]",
    )
    chunk_nist = RetrievedChunk(
        chunk_id="c2",
        document_id="nist-csf-20",
        text="Sample text",
        score=0.80,
        rank=2,
        source_label="[SOURCE_2]",
    )

    # OWASP check
    assert is_expected_source_retrieved([chunk_owasp], "owasp") is True
    assert is_expected_source_retrieved([chunk_nist], "owasp") is False

    # NIST check
    assert is_expected_source_retrieved([chunk_nist], "nist") is True
    assert is_expected_source_retrieved([chunk_owasp], "nist") is False

    # None / unsupported check
    assert is_expected_source_retrieved([chunk_owasp], "none") is True


def test_is_unsupported_refusal():
    """Test refusal indicator detection."""
    assert is_unsupported_refusal("The supplied documents do not contain enough information.") is True
    assert is_unsupported_refusal("Without access to the specific recipes mentioned in the evidence, I cannot provide.") is True
    assert is_unsupported_refusal("There is insufficient evidence to answer this.") is True
    assert is_unsupported_refusal("Multi-factor authentication is required by CISA.") is False


def test_evaluate_question_pair_and_aggregation():
    """Test evaluating a Base/RAG answer pair and computing aggregate metrics."""
    q = EvaluationQuestion(
        id="q1",
        question="What is broken access control?",
        expected_source="owasp",
        reference_points=["Broken Access Control", "privilege"],
    )

    base_res = AnswerResult(
        question=q.question,
        pipeline="base",
        answer="Broken access control occurs when users can act outside their privilege.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=1200.0,
        sources=[],
        metadata={"input_token_count": 50, "output_token_count": 80},
    )

    rag_chunk = RetrievedChunk(
        chunk_id="chk_1",
        document_id="a01-2025-broken-access-control",
        text="A01 covers Broken Access Control and privilege violation.",
        score=0.9,
        rank=1,
        source_label="[SOURCE_1]",
        metadata={"section_heading": "Broken Access Control"},
    )
    rag_res = AnswerResult(
        question=q.question,
        pipeline="rag",
        answer="According to [SOURCE_1], Broken Access Control involves privilege escalation.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=1500.0,
        sources=[rag_chunk],
        metadata={
            "retrieval_latency_ms": 50.0,
            "generation_latency_ms": 1450.0,
            "input_token_count": 500,
            "output_token_count": 90,
            "context_token_count": 400,
            "citation_labels_used": ["[SOURCE_1]"],
            "invalid_citations": [],
        },
    )

    pair = evaluate_question_pair(q, base_res, rag_res)
    assert pair["id"] == "q1"
    assert pair["base"]["reference_coverage"] == 1.0
    assert pair["rag"]["reference_coverage"] == 1.0
    assert pair["rag"]["expected_source_retrieved"] is True
    assert pair["rag"]["citation_validity"] == 1.0

    # Aggregate
    agg = aggregate_evaluation_results([pair])
    assert agg["total_questions"] == 1
    assert agg["base"]["avg_reference_coverage"] == 1.0
    assert agg["rag"]["avg_reference_coverage"] == 1.0
    assert agg["rag"]["expected_source_retrieval_rate"] == 1.0
    assert agg["rag"]["citation_validity"] == 1.0


def test_evaluation_result_json_serialization(tmp_path: Path):
    """Test that full evaluation dictionary can be cleanly dumped and reloaded as JSON."""
    q = EvaluationQuestion(
        id="q1",
        question="What is MFA?",
        expected_source="cisa",
        reference_points=["MFA"],
    )
    base_res = AnswerResult(
        question="What is MFA?",
        pipeline="base",
        answer="MFA is multi-factor authentication.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=100.0,
        sources=[],
    )
    rag_res = AnswerResult(
        question="What is MFA?",
        pipeline="rag",
        answer="MFA is multi-factor authentication.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=150.0,
        sources=[],
    )

    pair = evaluate_question_pair(q, base_res, rag_res)
    agg = aggregate_evaluation_results([pair])

    output = {
        "metadata": {"version": "v1"},
        "summary": agg,
        "items": [pair],
    }

    out_file = tmp_path / "eval_out.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["metadata"]["version"] == "v1"
    assert loaded["summary"]["total_questions"] == 1
    assert loaded["items"][0]["id"] == "q1"


# ── Phase 9 Unit Tests ───────────────────────────────────────────────────────


def test_phase9_derived_dataset_preserves_heldout_records():
    """Verify that phase9_final_test.jsonl is an exact 9-record structural projection of finetune_test.jsonl."""
    from contextforge.evaluation.evaluator import load_phase9_test_dataset

    derived_records = load_phase9_test_dataset("data/qa/phase9_final_test.jsonl")
    assert len(derived_records) == 9

    with open("data/qa/finetune_test.jsonl", "r", encoding="utf-8") as f:
        src_recs = [json.loads(line) for line in f if line.strip()]

    assert len(src_recs) == 9

    for d, s in zip(derived_records, src_recs):
        user_q = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
        asst_a = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]

        assert d.id == s["id"]
        assert d.question == user_q
        assert d.reference_answer == asst_a
        assert d.source_family == s["source_family"]
        assert d.source_document_id == s["source_document_id"]
        assert d.category == s["category"]


def test_compute_token_scores_deterministic():
    """Verify normalized token precision, recall, and F1 calculations."""
    from contextforge.evaluation.evaluator import compute_token_scores

    # Exact match
    p, r, f1 = compute_token_scores("Multi-factor authentication (MFA)", "Multi-factor authentication (MFA)")
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    # Partial overlap
    p, r, f1 = compute_token_scores("CISA recommends MFA.", "CISA recommends implementing strong MFA policies.")
    # pred tokens: ['cisa', 'recommends', 'mfa'] (3)
    # ref tokens: ['cisa', 'recommends', 'implementing', 'strong', 'mfa', 'policies'] (6)
    # overlap: 3
    assert p == 1.0
    assert r == 0.5
    assert f1 == round((2 * 1.0 * 0.5) / 1.5, 4)

    # Disjoint
    p, r, f1 = compute_token_scores("Apples and oranges", "Chocolate chip cookies")
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_compute_retrieval_metrics():
    """Verify hit@K and MRR computation against expected document ID."""
    from contextforge.evaluation.evaluator import compute_retrieval_metrics

    sources = [
        RetrievedChunk(chunk_id="c1", document_id="a02-2025-security-misconfiguration", text="", score=0.9, rank=1, source_label="[SOURCE_1]"),
        RetrievedChunk(chunk_id="c2", document_id="a05-2025-injection", text="", score=0.85, rank=2, source_label="[SOURCE_2]"),
        RetrievedChunk(chunk_id="c3", document_id="cisa-cpg-20", text="", score=0.8, rank=3, source_label="[SOURCE_3]"),
    ]

    # Target is rank 2
    metrics = compute_retrieval_metrics(sources, "owasp-top10-2025-a05")
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["first_match_rank"] == 2

    # Target not in sources
    metrics_miss = compute_retrieval_metrics(sources, "nist-csf-2.0")
    assert metrics_miss["hit_at_1"] == 0.0
    assert metrics_miss["hit_at_5"] == 0.0
    assert metrics_miss["mrr"] == 0.0
    assert metrics_miss["first_match_rank"] is None


def test_evaluate_phase9_triplet_and_aggregation():
    """Verify evaluate_phase9_triplet and aggregate_phase9_results produce expected structure."""
    from unittest.mock import MagicMock
    from contextforge.evaluation.evaluator import (
        Phase9TestQuestion,
        aggregate_phase9_results,
        evaluate_phase9_triplet,
    )

    q = Phase9TestQuestion(
        id="ft_0001",
        question="What is MFA?",
        reference_answer="Multi-factor authentication requires multiple credentials.",
        source_family="cisa",
        source_document_id="cisa-cpg-2.0",
        category="factual_definition",
    )

    base_res = AnswerResult(
        question="What is MFA?",
        pipeline="base",
        answer="MFA is multi-factor authentication.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=100.0,
        metadata={"input_token_count": 10, "output_token_count": 6},
    )
    rag_sources = [
        RetrievedChunk(chunk_id="c1", document_id="cisa-cpg-20", text="", score=0.9, rank=1, source_label="[SOURCE_1]")
    ]
    rag_res = AnswerResult(
        question="What is MFA?",
        pipeline="rag",
        answer="According to [SOURCE_1], Multi-factor authentication requires multiple credentials.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        sources=rag_sources,
        latency_ms=250.0,
        metadata={
            "input_token_count": 500,
            "output_token_count": 12,
            "context_token_count": 450,
            "retrieval_latency_ms": 30.0,
            "generation_latency_ms": 220.0,
            "citation_labels_used": ["[SOURCE_1]"],
            "invalid_citations": [],
        },
    )
    ft_res = AnswerResult(
        question="What is MFA?",
        pipeline="finetuned",
        answer="Multi-factor authentication requires multiple credentials.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        latency_ms=120.0,
        metadata={"input_token_count": 10, "output_token_count": 8},
    )


    # Mock embedder for fast unit test
    mock_embedder = MagicMock()
    mock_embedder.embed_passages.return_value = [[1.0, 0.0], [1.0, 0.0]]

    eval_item = evaluate_phase9_triplet(q, base_res, rag_res, ft_res, embedder=mock_embedder)
    assert eval_item["id"] == "ft_0001"
    assert "token_f1" in eval_item["base"]
    assert "token_f1" in eval_item["rag"]
    assert "token_f1" in eval_item["finetuned"]
    assert eval_item["rag"]["retrieval"]["hit_at_1"] == 1.0

    agg = aggregate_phase9_results([eval_item])
    assert agg["total_test_questions"] == 1
    assert agg["unsupported_refusal_metrics"] == "NOT_APPLICABLE"
    assert "token_f1_mean" in agg["base"]
    assert "token_f1_mean" in agg["rag"]
    assert "token_f1_mean" in agg["finetuned"]
    assert agg["rag"]["document_hit_at_1"] == 1.0
    assert agg["rag"]["citation_presence_rate"] == 1.0
    assert agg["rag"]["citation_validity"] == 1.0

