"""
ContextForge LLM Package.
=========================
Contains base causal language model loader, prompt templates, and generation wrappers.
"""

from __future__ import annotations

from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.generation import GenerationConfig, GenerationResult, TextGenerator
from contextforge.llm.prompts import BASE_PROMPT_V1, format_base_messages, get_base_system_prompt

__all__ = [
    "BASE_PROMPT_V1",
    "BaseModelLoader",
    "GenerationConfig",
    "GenerationResult",
    "TextGenerator",
    "format_base_messages",
    "get_base_system_prompt",
]
