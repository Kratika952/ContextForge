"""
Token-Aware Document Chunker for ContextForge.
==============================================
Segments extracted documents into token-bounded chunks with full provenance
using the canonical BAAI/bge-small-en-v1.5 tokenizer.

Implements Section 6.2 and Phase 3 requirements of IMPLEMENTATION_PLAN.md:
- Strict token counting via BGE tokenizer (never character/word heuristics).
- Preserves document ID, page ranges (PDF), section headings (Markdown), and metadata.
- Computes SHA-256 content hashes and deterministic chunk IDs.
- Guarantees zero empty chunks and non-violating token upper bounds.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from contextforge.common.types import (
    Chunk,
    Document,
    ExtractedDocument,
    ExtractedUnit,
)
from contextforge.ingestion.markdown_loader import MarkdownLoader
from contextforge.ingestion.pdf_loader import PDFLoader

# Approved experimental chunking profiles from IMPLEMENTATION_PLAN.md §10 Phase 3
DEFAULT_PROFILES: dict[str, dict[str, int]] = {
    "small": {
        "chunk_size_tokens": 180,
        "overlap_tokens": 30,
    },
    "medium": {
        "chunk_size_tokens": 300,
        "overlap_tokens": 50,
    },
    "large": {
        "chunk_size_tokens": 420,
        "overlap_tokens": 70,
    },
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class _TextAtom:
    """Internal atomic text slice (paragraph or sentence) with provenance."""

    text: str
    token_count: int
    page: int | None = None
    heading: str | None = None


class TokenAwareChunker:
    """Token-bounded text chunker using the BGE-small-en-v1.5 tokenizer."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        tokenizer: Any | None = None,
        profiles: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Initialize TokenAwareChunker.

        Args:
            model_name: HuggingFace model identifier for the tokenizer.
            tokenizer: Pre-instantiated HuggingFace tokenizer instance (optional).
            profiles: Dictionary mapping profile name to {chunk_size_tokens, overlap_tokens}.
        """
        self.model_name = model_name
        self.profiles = profiles or DEFAULT_PROFILES

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set high model_max_length on tokenizer to prevent spurious warnings
        # when counting tokens on large raw document paragraphs.
        self.tokenizer.model_max_length = 100_000

        self.pdf_loader = PDFLoader()
        self.markdown_loader = MarkdownLoader()

    def count_tokens(self, text: str) -> int:
        """Compute exact token count for text using the BGE tokenizer.

        Args:
            text: Input string.

        Returns:
            Exact number of tokens.
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_long_text(
        self,
        text: str,
        chunk_size_tokens: int,
        overlap_tokens: int,
        page: int | None,
        heading: str | None,
    ) -> list[_TextAtom]:
        """Subdivide text that exceeds chunk_size_tokens into token-bounded atoms."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        step = max(1, chunk_size_tokens - overlap_tokens)
        sub_atoms: list[_TextAtom] = []

        for i in range(0, len(tokens), step):
            slice_tokens = tokens[i : i + chunk_size_tokens]
            decoded = self.tokenizer.decode(slice_tokens, skip_special_tokens=True).strip()
            if not decoded:
                continue

            # Verify actual token count after decode
            tok_count = self.count_tokens(decoded)
            while tok_count > chunk_size_tokens and len(slice_tokens) > 1:
                # Trim one token at a time in the rare case subword re-encoding increases tokens
                slice_tokens = slice_tokens[:-1]
                decoded = self.tokenizer.decode(slice_tokens, skip_special_tokens=True).strip()
                tok_count = self.count_tokens(decoded)

            if decoded and tok_count > 0:
                sub_atoms.append(
                    _TextAtom(
                        text=decoded,
                        token_count=tok_count,
                        page=page,
                        heading=heading,
                    )
                )

        return sub_atoms

    def _extract_atoms(
        self,
        extracted: ExtractedDocument,
        chunk_size_tokens: int,
        overlap_tokens: int,
    ) -> list[_TextAtom]:
        """Segment an ExtractedDocument into an ordered sequence of token-bounded atoms."""
        atoms: list[_TextAtom] = []

        for unit in extracted.units:
            unit_text = unit.text.strip()
            if not unit_text:
                continue

            page = unit.page_number if extracted.source_type == "pdf" else None
            heading = unit.heading if extracted.source_type == "markdown" else None

            # Split unit into paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", unit_text) if p.strip()]

            for p in paragraphs:
                p_tokens = self.count_tokens(p)
                if p_tokens <= chunk_size_tokens:
                    atoms.append(_TextAtom(text=p, token_count=p_tokens, page=page, heading=heading))
                else:
                    # Paragraph exceeds limit: split into sentences
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                    for s in sentences:
                        s_tokens = self.count_tokens(s)
                        if s_tokens <= chunk_size_tokens:
                            atoms.append(
                                _TextAtom(text=s, token_count=s_tokens, page=page, heading=heading)
                            )
                        else:
                            # Sentence exceeds limit: subdivide token sequence
                            atoms.extend(
                                self._split_long_text(
                                    s, chunk_size_tokens, overlap_tokens, page, heading
                                )
                            )

        return atoms

    def chunk_document(
        self,
        doc: Document,
        extracted: ExtractedDocument | None = None,
        profile: str = "medium",
    ) -> list[Chunk]:
        """Segment a Document into a list of Chunk objects.

        Args:
            doc: Document metadata instance.
            extracted: Pre-extracted document (optional). If None, loaded from disk.
            profile: Chunking profile name ("small", "medium", or "large").

        Returns:
            List of Chunk instances with provenance and token counts.

        Raises:
            ValueError: If the profile is not recognized.
            FileNotFoundError: If the source document file cannot be found.
        """
        if profile not in self.profiles:
            raise ValueError(
                f"Unknown chunking profile '{profile}'. Allowed profiles: {list(self.profiles.keys())}"
            )

        cfg = self.profiles[profile]
        chunk_size_tokens = cfg["chunk_size_tokens"]
        overlap_tokens = cfg["overlap_tokens"]

        if extracted is None:
            source_path = Path(doc.source_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Source file not found for chunking: {source_path}")

            if doc.source_type == "pdf":
                extracted = self.pdf_loader.load(source_path)
            elif doc.source_type == "markdown":
                extracted = self.markdown_loader.load(source_path)
            else:
                raise ValueError(f"Unsupported source type '{doc.source_type}' for chunking.")

        atoms = self._extract_atoms(extracted, chunk_size_tokens, overlap_tokens)
        if not atoms:
            return []

        chunks: list[Chunk] = []
        chunk_idx = 0
        current_start = 0

        while current_start < len(atoms):
            current_atoms: list[_TextAtom] = []
            end_idx = current_start

            while end_idx < len(atoms):
                candidate_atom = atoms[end_idx]
                if not current_atoms:
                    # Always include the first atom (already bounded <= chunk_size_tokens)
                    current_atoms.append(candidate_atom)
                    end_idx += 1
                else:
                    candidate_text = "\n\n".join(a.text for a in (current_atoms + [candidate_atom]))
                    candidate_tokens = self.count_tokens(candidate_text)
                    if candidate_tokens <= chunk_size_tokens:
                        current_atoms.append(candidate_atom)
                        end_idx += 1
                    else:
                        break

            chunk_text = "\n\n".join(a.text for a in current_atoms).strip()
            if chunk_text:
                pages = [a.page for a in current_atoms if a.page is not None]
                headings = [a.heading for a in current_atoms if a.heading is not None]

                page_start = min(pages) if pages else None
                page_end = max(pages) if pages else None
                section_heading = headings[0] if headings else None

                content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                actual_tokens = self.count_tokens(chunk_text)
                chunk_id = f"{doc.id}_{profile}_{chunk_idx:04d}"

                chunk_metadata = {
                    "document_title": doc.title,
                    "source_type": doc.source_type,
                    "source_path": doc.source_path,
                }
                if doc.metadata:
                    chunk_metadata["document_metadata"] = doc.metadata

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        chunk_index=chunk_idx,
                        chunk_profile=profile,
                        text=chunk_text,
                        token_count=actual_tokens,
                        content_hash=content_hash,
                        page_start=page_start,
                        page_end=page_end,
                        section_heading=section_heading,
                        metadata=chunk_metadata,
                    )
                )
                chunk_idx += 1

            if end_idx >= len(atoms):
                break

            # Calculate overlap start position
            overlap_start = end_idx - 1
            while overlap_start > current_start:
                overlap_candidate = atoms[overlap_start:end_idx]
                overlap_text = "\n\n".join(a.text for a in overlap_candidate)
                if self.count_tokens(overlap_text) > overlap_tokens:
                    overlap_start += 1
                    break
                overlap_start -= 1

            # Ensure strict forward progress
            current_start = max(current_start + 1, overlap_start)

        return chunks

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc",
        profile: str = "medium",
        title: str = "Text Document",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Convenience method to chunk raw text directly.

        Args:
            text: Raw input text.
            document_id: Identifier to assign to chunks.
            profile: Chunking profile.
            title: Document title.
            metadata: Optional additional metadata dict.

        Returns:
            List of Chunk instances.
        """
        doc = Document(
            id=document_id,
            title=title,
            source_path=":raw_text:",
            checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            source_type="markdown",
            status="INGESTED",
            metadata=metadata or {},
        )
        extracted = ExtractedDocument(
            source_path=":raw_text:",
            checksum=doc.checksum,
            source_type="markdown",
            units=[ExtractedUnit(unit_index=1, text=text, char_count=len(text), heading=None)],
            total_units=1,
            title=title,
        )
        return self.chunk_document(doc, extracted=extracted, profile=profile)
