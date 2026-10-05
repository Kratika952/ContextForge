"""
Evaluation module for ContextForge: Base vs RAG evaluation harness.
"""

from __future__ import annotations

from contextforge.evaluation.evaluator import (
    EvaluationQuestion,
    aggregate_evaluation_results,
    calculate_reference_coverage,
    evaluate_question_pair,
    is_expected_source_retrieved,
    is_unsupported_refusal,
    load_evaluation_dataset,
)

__all__ = [
    "EvaluationQuestion",
    "aggregate_evaluation_results",
    "calculate_reference_coverage",
    "evaluate_question_pair",
    "is_expected_source_retrieved",
    "is_unsupported_refusal",
    "load_evaluation_dataset",
]
