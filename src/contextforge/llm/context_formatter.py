"""
Context Formatter for ContextForge RAG Pipeline.
===============================================
Deterministically formats retrieved evidence chunks with standardized source labels,
human-readable provenance, and token budget enforcement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from contextforge.common.types import RetrievedChunk

DEFAULT_MAX_CONTEXT_TOKENS = 1500


@dataclass
class FormattedContext:
    """Structured result of evidence context formatting."""

    context_text: str
    included_chunks: list[RetrievedChunk] = field(default_factory=list)
    label_to_chunk: dict[str, RetrievedChunk] = field(default_factory=dict)
    context_token_count: int = 0
    retrieved_count: int = 0
    included_count: int = 0


def resolve_document_source_name(document_id: str, metadata: dict[str, Any] | None = None) -> str:
    """Resolve human-readable source title from document_id or metadata.

    Args:
        document_id: Unique document identifier.
        metadata: Optional metadata dictionary.

    Returns:
        Human-readable authority/corpus title.
    """
    if metadata and metadata.get("title"):
        return str(metadata["title"])

    doc_lower = document_id.lower()
    if doc_lower.startswith("nist-csf"):
        return "NIST CSF 2.0"
    if doc_lower.startswith("cisa-cpg"):
        return "CISA CPG 2.0"
    if "owasp" in doc_lower or doc_lower.startswith("a0") or doc_lower.startswith("0"):
        return "OWASP Top 10:2025"

    return document_id


def resolve_reference_label(chunk: RetrievedChunk) -> str:
    """Resolve human-readable section or page reference for a retrieved chunk.

    Args:
        chunk: RetrievedChunk instance.

    Returns:
        Reference string (e.g. 'Page 12', 'A01 Broken Access Control — How to Prevent').
    """
    page_start = chunk.metadata.get("page_start")
    page_end = chunk.metadata.get("page_end")

    if page_start is not None:
        if page_end is not None and page_end != page_start:
            return f"Pages {page_start}–{page_end}"
        return f"Page {page_start}"

    heading = chunk.metadata.get("section_heading")
    if heading:
        clean_heading = re.sub(r"^[#\s]+", "", heading).strip()
        return clean_heading

    return chunk.source_label


class ContextFormatter:
    """Formats retrieved chunks into deterministic context blocks with token budgeting."""

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        tokenizer: Any | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize ContextFormatter.

        Args:
            max_context_tokens: Maximum token budget allocated for retrieved evidence.
            tokenizer: Hugging Face tokenizer instance for token counting.
            token_counter: Optional custom callable (text -> int) for token counting.
        """
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer
        self.token_counter = token_counter

    def count_tokens(self, text: str) -> int:
        """Count tokens in text string using configured tokenizer or fallback estimator.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        if self.token_counter is not None:
            return self.token_counter(text)
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass
        # Conservative word-based estimate (~1.33 tokens per whitespace-delimited word)
        words = text.split()
        return max(1, int(len(words) * 1.33))

    def format_single_chunk(self, label: str, chunk: RetrievedChunk) -> str:
        """Format a single passage block.

        Args:
            label: Prompt citation label (e.g. 'SOURCE_1').
            chunk: RetrievedChunk instance.

        Returns:
            Formatted passage string.
        """
        source_name = resolve_document_source_name(chunk.document_id, chunk.metadata)
        reference = resolve_reference_label(chunk)

        return (
            f"[{label}]\n"
            f"Source: {source_name}\n"
            f"Reference: {reference}\n"
            f"Content:\n"
            f"{chunk.text.strip()}"
        )

    def format(self, chunks: list[RetrievedChunk]) -> FormattedContext:
        """Deterministically format chunks into structured evidence adhering to token budget.

        Retrieved chunks are evaluated in order. If adding the next chunk would exceed
        max_context_tokens, the chunk is skipped (no mid-passage truncation).

        Args:
            chunks: Ranked list of RetrievedChunk instances.

        Returns:
            FormattedContext containing formatted string, mappings, and token usage.
        """
        if not chunks:
            return FormattedContext(
                context_text="",
                included_chunks=[],
                label_to_chunk={},
                context_token_count=0,
                retrieved_count=0,
                included_count=0,
            )

        retrieved_count = len(chunks)
        included_chunks: list[RetrievedChunk] = []
        label_to_chunk: dict[str, RetrievedChunk] = {}
        formatted_blocks: list[str] = []

        current_token_count = 0

        for i, chunk in enumerate(chunks, start=1):
            label = f"SOURCE_{i}"
            block_text = self.format_single_chunk(label, chunk)
            block_tokens = self.count_tokens(block_text)

            # Check token budget before including
            if current_token_count + block_tokens > self.max_context_tokens:
                # Token budget exceeded: skip next chunk to prevent partial/truncated evidence
                continue

            formatted_blocks.append(block_text)
            included_chunks.append(chunk)
            label_to_chunk[label] = chunk
            current_token_count += block_tokens

        full_context_text = "\n\n".join(formatted_blocks)
        final_token_count = self.count_tokens(full_context_text)

        return FormattedContext(
            context_text=full_context_text,
            included_chunks=included_chunks,
            label_to_chunk=label_to_chunk,
            context_token_count=final_token_count,
            retrieved_count=retrieved_count,
            included_count=len(included_chunks),
        )
