"""
ContextForge LLM Package.
=========================
Contains base causal language model loader, prompt templates, and generation wrappers.
"""

from __future__ import annotations

from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.citations import CitationValidationResult, extract_citation_labels, validate_citations
from contextforge.llm.context_formatter import ContextFormatter, FormattedContext
from contextforge.llm.generation import GenerationConfig, GenerationResult, TextGenerator
from contextforge.llm.prompts import (
    BASE_PROMPT_V1,
    RAG_PROMPT_V1,
    format_base_messages,
    format_rag_messages,
    get_base_system_prompt,
    get_rag_system_prompt,
)

__all__ = [
    "BASE_PROMPT_V1",
    "BaseModelLoader",
    "CitationValidationResult",
    "ContextFormatter",
    "FormattedContext",
    "GenerationConfig",
    "GenerationResult",
    "RAG_PROMPT_V1",
    "TextGenerator",
    "extract_citation_labels",
    "format_base_messages",
    "format_rag_messages",
    "get_base_system_prompt",
    "get_rag_system_prompt",
    "validate_citations",
]

