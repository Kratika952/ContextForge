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


# ── Phase 9 — Final Three-Way Comparative Evaluation ─────────────────────────


@dataclass
class Phase9TestQuestion:
    """Held-out Phase 9 test question record."""

    id: str
    question: str
    reference_answer: str
    source_family: str
    source_document_id: str
    category: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase9TestQuestion:
        """Parse Phase9TestQuestion from raw dictionary."""
        for field_name in ("id", "question", "reference_answer", "source_family", "source_document_id", "category"):
            if not data.get(field_name):
                raise ValueError(f"Phase 9 record missing required '{field_name}' field.")

        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            reference_answer=str(data["reference_answer"]),
            source_family=str(data["source_family"]),
            source_document_id=str(data["source_document_id"]),
            category=str(data["category"]),
        )


def load_phase9_test_dataset(path: Path | str) -> list[Phase9TestQuestion]:
    """Load Phase 9 held-out test questions from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Phase 9 dataset file not found at: {p}")

    questions: list[Phase9TestQuestion] = []
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {idx} in {p}: {err}") from err
            questions.append(Phase9TestQuestion.from_dict(data))

    if not questions:
        raise ValueError(f"No Phase 9 test records found in {p}")

    return questions


def compute_token_scores(prediction: str, reference: str) -> tuple[float, float, float]:
    """Compute normalized SQuAD-style token precision, recall, and F1.

    Returns:
        (precision, recall, f1) rounded to 4 decimals.
    """
    import string
    from collections import Counter

    def _normalize(text: str) -> list[str]:
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        return text.split()

    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)

    if not pred_tokens or not ref_tokens:
        if not pred_tokens and not ref_tokens:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    embedder: Any | None = None,
) -> float:
    """Compute cosine semantic similarity between prediction and reference answer.

    Uses normalized embeddings from the approved BAAI/bge-small-en-v1.5 model.
    """
    import numpy as np

    if not prediction or not reference:
        return 0.0

    if embedder is None:
        from contextforge.retrieval.embedder import BGEEmbedder
        embedder = BGEEmbedder()

    vecs = embedder.embed_passages([prediction, reference])
    sim = float(np.dot(vecs[0], vecs[1]))
    return round(float(np.clip(sim, -1.0, 1.0)), 4)


def is_document_match(chunk_doc_id: str, expected_doc_id: str) -> bool:
    """Match chunk document ID against expected corpus manifest document ID."""
    c = chunk_doc_id.lower().replace("-", "").replace(".", "").replace("_", "")
    e = expected_doc_id.lower().replace("-", "").replace(".", "").replace("_", "")

    if c == e or e in c or c in e:
        return True

    # OWASP category cross-matching (e.g. owasptop102025a05 vs a052025injection)
    for cat in ("a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", "intro"):
        if cat in e and cat in c:
            return True

    return False


def compute_retrieval_metrics(
    sources: list[RetrievedChunk],
    expected_document_id: str,
) -> dict[str, Any]:
    """Compute RAG retrieval metrics: hit@1, hit@3, hit@5, and MRR."""
    hit_1 = 0.0
    hit_3 = 0.0
    hit_5 = 0.0
    mrr = 0.0

    matching_ranks: list[int] = []
    for rank, chunk in enumerate(sources, start=1):
        if is_document_match(chunk.document_id, expected_document_id):
            matching_ranks.append(rank)

    if matching_ranks:
        first_rank = matching_ranks[0]
        hit_1 = 1.0 if first_rank <= 1 else 0.0
        hit_3 = 1.0 if first_rank <= 3 else 0.0
        hit_5 = 1.0 if first_rank <= 5 else 0.0
        mrr = round(1.0 / first_rank, 4)

    return {
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "hit_at_5": hit_5,
        "mrr": mrr,
        "first_match_rank": matching_ranks[0] if matching_ranks else None,
    }


def evaluate_phase9_triplet(
    question: Phase9TestQuestion,
    base_result: AnswerResult,
    rag_result: AnswerResult,
    finetuned_result: AnswerResult,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Evaluate Base, RAG, and Fine-Tuned answers for a single Phase 9 test record."""
    ref = question.reference_answer

    # Base quality & telemetry
    b_p, b_r, b_f1 = compute_token_scores(base_result.answer, ref)
    b_sim = compute_semantic_similarity(base_result.answer, ref, embedder=embedder)
    b_meta = base_result.metadata or {}

    # RAG quality & telemetry
    r_p, r_r, r_f1 = compute_token_scores(rag_result.answer, ref)
    r_sim = compute_semantic_similarity(rag_result.answer, ref, embedder=embedder)
    r_meta = rag_result.metadata or {}

    # Fine-Tuned quality & telemetry
    ft_p, ft_r, ft_f1 = compute_token_scores(finetuned_result.answer, ref)
    ft_sim = compute_semantic_similarity(finetuned_result.answer, ref, embedder=embedder)
    ft_meta = finetuned_result.metadata or {}

    # RAG retrieval metrics
    ret_metrics = compute_retrieval_metrics(rag_result.sources, question.source_document_id)

    # Citation metrics
    cited_labels = r_meta.get("citation_labels_used", [])
    invalid_labels = r_meta.get("invalid_citations", [])
    total_cited = len(cited_labels)
    valid_cited = total_cited - len(invalid_labels)
    has_citations = total_cited > 0
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
        "reference_answer": ref,
        "source_family": question.source_family,
        "source_document_id": question.source_document_id,
        "category": question.category,
        "base": {
            "answer": base_result.answer,
            "latency_ms": round(base_result.latency_ms, 2),
            "input_tokens": b_meta.get("input_token_count", 0),
            "output_tokens": b_meta.get("output_token_count", 0),
            "token_precision": b_p,
            "token_recall": b_r,
            "token_f1": b_f1,
            "semantic_similarity": b_sim,
        },
        "rag": {
            "answer": rag_result.answer,
            "latency_ms": round(rag_result.latency_ms, 2),
            "retrieval_latency_ms": round(r_meta.get("retrieval_latency_ms", 0.0), 2),
            "generation_latency_ms": round(r_meta.get("generation_latency_ms", 0.0), 2),
            "input_tokens": r_meta.get("input_token_count", 0),
            "output_tokens": r_meta.get("output_token_count", 0),
            "context_tokens": r_meta.get("context_token_count", 0),
            "token_precision": r_p,
            "token_recall": r_r,
            "token_f1": r_f1,
            "semantic_similarity": r_sim,
            "retrieval": ret_metrics,
            "has_citations": has_citations,
            "citations_used": cited_labels,
            "invalid_citations": invalid_labels,
            "citation_validity": round(citation_validity, 4),
            "sources": sources_summary,
        },
        "finetuned": {
            "answer": finetuned_result.answer,
            "latency_ms": round(finetuned_result.latency_ms, 2),
            "input_tokens": ft_meta.get("input_token_count", 0),
            "output_tokens": ft_meta.get("output_token_count", 0),
            "token_precision": ft_p,
            "token_recall": ft_r,
            "token_f1": ft_f1,
            "semantic_similarity": ft_sim,
        },
    }


def aggregate_phase9_results(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate individual Phase 9 test evaluations into three-way comparison metrics."""
    if not item_results:
        return {}

    n = len(item_results)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _median(values: list[float]) -> float:
        s = sorted(values)
        mid = len(s) // 2
        return (s[mid] if len(s) % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0) if s else 0.0

    # Base aggregates
    b_p = [r["base"]["token_precision"] for r in item_results]
    b_r = [r["base"]["token_recall"] for r in item_results]
    b_f1 = [r["base"]["token_f1"] for r in item_results]
    b_sim = [r["base"]["semantic_similarity"] for r in item_results]
    b_lat = [r["base"]["latency_ms"] for r in item_results]
    b_in_tok = [r["base"]["input_tokens"] for r in item_results]
    b_out_tok = [r["base"]["output_tokens"] for r in item_results]

    # RAG aggregates
    r_p = [r["rag"]["token_precision"] for r in item_results]
    r_r = [r["rag"]["token_recall"] for r in item_results]
    r_f1 = [r["rag"]["token_f1"] for r in item_results]
    r_sim = [r["rag"]["semantic_similarity"] for r in item_results]
    r_lat = [r["rag"]["latency_ms"] for r in item_results]
    r_ret_lat = [r["rag"]["retrieval_latency_ms"] for r in item_results]
    r_gen_lat = [r["rag"]["generation_latency_ms"] for r in item_results]
    r_in_tok = [r["rag"]["input_tokens"] for r in item_results]
    r_out_tok = [r["rag"]["output_tokens"] for r in item_results]
    r_ctx_tok = [r["rag"]["context_tokens"] for r in item_results]

    r_hit1 = [r["rag"]["retrieval"]["hit_at_1"] for r in item_results]
    r_hit3 = [r["rag"]["retrieval"]["hit_at_3"] for r in item_results]
    r_hit5 = [r["rag"]["retrieval"]["hit_at_5"] for r in item_results]
    r_mrr = [r["rag"]["retrieval"]["mrr"] for r in item_results]

    cited_questions = [r for r in item_results if r["rag"]["has_citations"]]
    citation_presence_rate = len(cited_questions) / n
    citation_validity_avg = (
        _mean([r["rag"]["citation_validity"] for r in cited_questions])
        if cited_questions
        else 1.0
    )

    # Fine-Tuned aggregates
    ft_p = [r["finetuned"]["token_precision"] for r in item_results]
    ft_r = [r["finetuned"]["token_recall"] for r in item_results]
    ft_f1 = [r["finetuned"]["token_f1"] for r in item_results]
    ft_sim = [r["finetuned"]["semantic_similarity"] for r in item_results]
    ft_lat = [r["finetuned"]["latency_ms"] for r in item_results]
    ft_in_tok = [r["finetuned"]["input_tokens"] for r in item_results]
    ft_out_tok = [r["finetuned"]["output_tokens"] for r in item_results]

    return {
        "total_test_questions": n,
        "unsupported_refusal_metrics": "NOT_APPLICABLE",
        "base": {
            "token_precision_mean": round(_mean(b_p), 4),
            "token_recall_mean": round(_mean(b_r), 4),
            "token_f1_mean": round(_mean(b_f1), 4),
            "semantic_similarity_mean": round(_mean(b_sim), 4),
            "latency_ms_mean": round(_mean(b_lat), 1),
            "latency_ms_median": round(_median(b_lat), 1),
            "input_tokens_mean": round(_mean(b_in_tok), 1),
            "output_tokens_mean": round(_mean(b_out_tok), 1),
        },
        "rag": {
            "token_precision_mean": round(_mean(r_p), 4),
            "token_recall_mean": round(_mean(r_r), 4),
            "token_f1_mean": round(_mean(r_f1), 4),
            "semantic_similarity_mean": round(_mean(r_sim), 4),
            "total_latency_ms_mean": round(_mean(r_lat), 1),
            "total_latency_ms_median": round(_median(r_lat), 1),
            "retrieval_latency_ms_mean": round(_mean(r_ret_lat), 1),
            "generation_latency_ms_mean": round(_mean(r_gen_lat), 1),
            "input_tokens_mean": round(_mean(r_in_tok), 1),
            "output_tokens_mean": round(_mean(r_out_tok), 1),
            "context_tokens_mean": round(_mean(r_ctx_tok), 1),
            "document_hit_at_1": round(_mean(r_hit1), 4),
            "document_hit_at_3": round(_mean(r_hit3), 4),
            "document_hit_at_5": round(_mean(r_hit5), 4),
            "document_mrr": round(_mean(r_mrr), 4),
            "citation_presence_rate": round(citation_presence_rate, 4),
            "citation_validity": round(citation_validity_avg, 4),
        },
        "finetuned": {
            "token_precision_mean": round(_mean(ft_p), 4),
            "token_recall_mean": round(_mean(ft_r), 4),
            "token_f1_mean": round(_mean(ft_f1), 4),
            "semantic_similarity_mean": round(_mean(ft_sim), 4),
            "latency_ms_mean": round(_mean(ft_lat), 1),
            "latency_ms_median": round(_median(ft_lat), 1),
            "input_tokens_mean": round(_mean(ft_in_tok), 1),
            "output_tokens_mean": round(_mean(ft_out_tok), 1),
        },
    }

