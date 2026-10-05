"""
Evaluation Harness for ContextForge: Base vs RAG Comparison.
============================================================
Provides deterministic, explainable metrics comparing BasePipeline vs RagPipeline:
- Reference-point coverage (rule-based factual criteria verification)
- Citation validity and citation frequency
- Expected source retrieval accuracy
- Unsupported domain refusal behavior
- Latency and token consumption telemetry

Avoids expensive LLM-as-a-judge dependencies and maintains fully reproducible evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from contextforge.common.types import AnswerResult, RetrievedChunk

REFUSAL_INDICATORS = [
    "not contain enough information",
    "does not contain enough information",
    "do not provide enough information",
    "cannot provide",
    "cannot answer",
    "insufficient evidence",
    "not enough information",
    "without access to",
    "no information",
    "not mentioned in the evidence",
    "not mentioned in the provided evidence",
]


@dataclass
class EvaluationQuestion:
    """Evaluation dataset record specification."""

    id: str
    question: str
    expected_source: str
    reference_points: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationQuestion:
        """Parse EvaluationQuestion from raw dictionary.

        Args:
            data: Dictionary containing question metadata.

        Returns:
            Parsed EvaluationQuestion instance.
        """
        if not data.get("id"):
            raise ValueError("Evaluation record missing required 'id' field.")
        if not data.get("question"):
            raise ValueError("Evaluation record missing required 'question' field.")
        if not data.get("expected_source"):
            raise ValueError("Evaluation record missing required 'expected_source' field.")

        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_source=str(data["expected_source"]).strip().lower(),
            reference_points=[str(p) for p in data.get("reference_points", [])],
        )


def load_evaluation_dataset(path: Path | str) -> list[EvaluationQuestion]:
    """Load evaluation questions from a JSONL file.

    Args:
        path: Path to .jsonl file.

    Returns:
        List of parsed EvaluationQuestion objects.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file is empty or lines are invalid JSON.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found at: {p}")

    questions: list[EvaluationQuestion] = []
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {idx} in {p}: {err}") from err
            questions.append(EvaluationQuestion.from_dict(data))

    if not questions:
        raise ValueError(f"No evaluation records found in {p}")

    return questions


def calculate_reference_coverage(answer: str, reference_points: list[str]) -> float:
    """Calculate rule-based coverage score of reference points within generated answer.

    Uses case-insensitive substring matching.

    Args:
        answer: Generated text response.
        reference_points: List of expected factual terms or concepts.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not reference_points:
        return 1.0
    if not answer:
        return 0.0

    ans_lower = answer.lower()
    matches = sum(1 for pt in reference_points if pt.lower() in ans_lower)
    return round(matches / len(reference_points), 4)


def is_expected_source_retrieved(sources: list[RetrievedChunk], expected_source: str) -> bool:
    """Verify if at least one retrieved chunk originates from the expected source family.

    Args:
        sources: List of RetrievedChunk instances.
        expected_source: Expected authority category ('owasp', 'nist', 'cisa', 'none').

    Returns:
        True if matching document or category found, or True if expected_source == 'none'.
    """
    exp = expected_source.strip().lower()
    if exp in ("", "none"):
        return True

    for chunk in sources:
        doc_lower = chunk.document_id.lower()
        if exp == "owasp" and ("owasp" in doc_lower or doc_lower.startswith("a0") or doc_lower.startswith("0")):
            return True
        if exp == "nist" and "nist" in doc_lower:
            return True
        if exp == "cisa" and "cisa" in doc_lower:
            return True
        if exp in doc_lower:
            return True

    return False


def is_unsupported_refusal(answer: str) -> bool:
    """Check if the answer text indicates insufficient evidence or proper refusal.

    Args:
        answer: Generated text response.

    Returns:
        True if refusal phrase is present.
    """
    if not answer:
        return False
    ans_lower = answer.lower()
    return any(phrase in ans_lower for phrase in REFUSAL_INDICATORS)


def evaluate_question_pair(
    question: EvaluationQuestion,
    base_result: AnswerResult,
    rag_result: AnswerResult,
) -> dict[str, Any]:
    """Evaluate Base and RAG answers for a single question record.

    Args:
        question: EvaluationQuestion specification.
        base_result: AnswerResult from BasePipeline.
        rag_result: AnswerResult from RagPipeline.

    Returns:
        Structured dictionary containing raw answers and deterministic metrics.
    """
    # Base metrics
    base_cov = calculate_reference_coverage(base_result.answer, question.reference_points)
    base_meta = base_result.metadata or {}

    # RAG metrics
    rag_cov = calculate_reference_coverage(rag_result.answer, question.reference_points)
    rag_meta = rag_result.metadata or {}

    expected_retrieved = is_expected_source_retrieved(rag_result.sources, question.expected_source)
    is_refusal = is_unsupported_refusal(rag_result.answer)

    cited_labels = rag_meta.get("citation_labels_used", [])
    invalid_labels = rag_meta.get("invalid_citations", [])
    total_cited = len(cited_labels)
    valid_cited = total_cited - len(invalid_labels)
    citation_validity = (valid_cited / total_cited) if total_cited > 0 else 1.0

    sources_summary = [
        {
            "label": src.source_label,
            "document_id": src.document_id,
            "score": round(src.score, 4),
            "reference": src.metadata.get("section_heading")
            or f"Page {src.metadata.get('page_start', '')}",
        }
        for src in rag_result.sources
    ]

    return {
        "id": question.id,
        "question": question.question,
        "expected_source": question.expected_source,
        "reference_points": question.reference_points,
        "base": {
            "answer": base_result.answer,
            "latency_ms": round(base_result.latency_ms, 2),
            "input_tokens": base_meta.get("input_token_count", 0),
            "output_tokens": base_meta.get("output_token_count", 0),
            "reference_coverage": base_cov,
        },
        "rag": {
            "answer": rag_result.answer,
            "latency_ms": round(rag_result.latency_ms, 2),
            "retrieval_latency_ms": round(rag_meta.get("retrieval_latency_ms", 0.0), 2),
            "generation_latency_ms": round(rag_meta.get("generation_latency_ms", 0.0), 2),
            "input_tokens": rag_meta.get("input_token_count", 0),
            "output_tokens": rag_meta.get("output_token_count", 0),
            "context_tokens": rag_meta.get("context_token_count", 0),
            "reference_coverage": rag_cov,
            "expected_source_retrieved": expected_retrieved,
            "citations_used": cited_labels,
            "invalid_citations": invalid_labels,
            "citation_validity": round(citation_validity, 4),
            "is_refusal": is_refusal,
            "sources": sources_summary,
        },
    }


def aggregate_evaluation_results(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate individual question evaluations into summary comparison metrics.

    Args:
        item_results: List of per-question evaluation dictionaries.

    Returns:
        Dictionary of aggregated comparison metrics.
    """
    if not item_results:
        return {}

    n = len(item_results)

    # Base aggregates
    base_cov_avg = sum(r["base"]["reference_coverage"] for r in item_results) / n
    base_latency_avg = sum(r["base"]["latency_ms"] for r in item_results) / n
    base_in_tokens_avg = sum(r["base"]["input_tokens"] for r in item_results) / n
    base_out_tokens_avg = sum(r["base"]["output_tokens"] for r in item_results) / n

    # RAG aggregates
    rag_cov_avg = sum(r["rag"]["reference_coverage"] for r in item_results) / n
    rag_latency_avg = sum(r["rag"]["latency_ms"] for r in item_results) / n
    rag_ret_latency_avg = sum(r["rag"]["retrieval_latency_ms"] for r in item_results) / n
    rag_gen_latency_avg = sum(r["rag"]["generation_latency_ms"] for r in item_results) / n
    rag_in_tokens_avg = sum(r["rag"]["input_tokens"] for r in item_results) / n
    rag_out_tokens_avg = sum(r["rag"]["output_tokens"] for r in item_results) / n
    rag_ctx_tokens_avg = sum(r["rag"]["context_tokens"] for r in item_results) / n

    # Source correctness (only on questions that target a corpus source)
    corpus_questions = [r for r in item_results if r["expected_source"] != "none"]
    if corpus_questions:
        expected_source_rate = (
            sum(1 for r in corpus_questions if r["rag"]["expected_source_retrieved"])
            / len(corpus_questions)
        )
    else:
        expected_source_rate = 1.0

    # Unsupported refusal rate (only on questions with expected_source == "none")
    unsupported_questions = [r for r in item_results if r["expected_source"] == "none"]
    if unsupported_questions:
        refusal_rate = (
            sum(1 for r in unsupported_questions if r["rag"]["is_refusal"])
            / len(unsupported_questions)
        )
    else:
        refusal_rate = 1.0

    # Citation validity across questions that cited
    questions_with_citations = [r for r in item_results if r["rag"]["citations_used"]]
    citation_frequency = len(questions_with_citations) / n
    if questions_with_citations:
        citation_validity_avg = (
            sum(r["rag"]["citation_validity"] for r in questions_with_citations)
            / len(questions_with_citations)
        )
    else:
        citation_validity_avg = 1.0

    return {
        "total_questions": n,
        "base": {
            "avg_reference_coverage": round(base_cov_avg, 4),
            "avg_latency_ms": round(base_latency_avg, 2),
            "avg_input_tokens": round(base_in_tokens_avg, 1),
            "avg_output_tokens": round(base_out_tokens_avg, 1),
        },
        "rag": {
            "avg_reference_coverage": round(rag_cov_avg, 4),
            "avg_total_latency_ms": round(rag_latency_avg, 2),
            "avg_retrieval_latency_ms": round(rag_ret_latency_avg, 2),
            "avg_generation_latency_ms": round(rag_gen_latency_avg, 2),
            "avg_input_tokens": round(rag_in_tokens_avg, 1),
            "avg_output_tokens": round(rag_out_tokens_avg, 1),
            "avg_context_tokens": round(rag_ctx_tokens_avg, 1),
            "expected_source_retrieval_rate": round(expected_source_rate, 4),
            "citation_frequency": round(citation_frequency, 4),
            "citation_validity": round(citation_validity_avg, 4),
            "unsupported_refusal_rate": round(refusal_rate, 4),
        },
    }
