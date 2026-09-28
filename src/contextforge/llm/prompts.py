"""
Prompt Templates for ContextForge LLM Pipelines.
================================================
Centralized, versioned prompt definitions.
"""

from __future__ import annotations

# Phase 4 Baseline prompt — minimal, direct, defensive cybersecurity persona.
# Does NOT claim access to supplied documents or mention retrieval mechanisms.
BASE_PROMPT_V1 = (
    "You are ContextForge, a defensive cybersecurity knowledge assistant.\n"
    "Answer the user's question clearly and concisely.\n"
    "If you are uncertain, say so rather than inventing information."
)

PROMPT_REGISTRY: dict[str, str] = {
    "base_v1": BASE_PROMPT_V1,
}


def get_base_system_prompt(version: str = "base_v1") -> str:
    """Retrieve the base system prompt for a given version.

    Args:
        version: Version string (default: 'base_v1').

    Returns:
        System prompt string.

    Raises:
        ValueError: If version is unknown.
    """
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt version '{version}'. Available versions: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[version]


def format_base_messages(question: str, prompt_version: str = "base_v1") -> list[dict[str, str]]:
    """Format standard chat messages for baseline question answering.

    Args:
        question: User query string.
        prompt_version: Version identifier for system prompt.

    Returns:
        List of role/content message dictionaries compatible with tokenizer chat templates.
    """
    system_content = get_base_system_prompt(prompt_version)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question.strip()},
    ]
