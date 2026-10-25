"""
Evaluation module for ContextForge: Base vs RAG evaluation harness.
"""

from __future__ import annotations

from contextforge.evaluation.evaluator import (
    EvaluationQuestion,
    Phase9TestQuestion,
    aggregate_evaluation_results,
    aggregate_phase9_results,
    calculate_reference_coverage,
    compute_retrieval_metrics,
    compute_semantic_similarity,
    compute_token_scores,
    evaluate_phase9_triplet,
    evaluate_question_pair,
    is_document_match,
    is_expected_source_retrieved,
    is_unsupported_refusal,
    load_evaluation_dataset,
    load_phase9_test_dataset,
)

__all__ = [
    "EvaluationQuestion",
    "Phase9TestQuestion",
    "aggregate_evaluation_results",
    "aggregate_phase9_results",
    "calculate_reference_coverage",
    "compute_retrieval_metrics",
    "compute_semantic_similarity",
    "compute_token_scores",
    "evaluate_phase9_triplet",
    "evaluate_question_pair",
    "is_document_match",
    "is_expected_source_retrieved",
    "is_unsupported_refusal",
    "load_evaluation_dataset",
    "load_phase9_test_dataset",
]


