"""
RAG Pipeline for ContextForge.
==============================
Executes evidence-grounded domain question answering by orchestrating dense retrieval,
deterministic context formatting, prompt injection defense, and causal text generation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from contextforge.common.config import get_db_path
from contextforge.common.types import (
    AnswerResult,
    IndexNotFoundError,
    InvalidQuestionError,
    RetrievedChunk,
)
from contextforge.llm.base_model import BaseModelLoader
from contextforge.llm.citations import CitationValidationResult, validate_citations
from contextforge.llm.context_formatter import ContextFormatter, FormattedContext
from contextforge.llm.generation import GenerationConfig, TextGenerator
from contextforge.llm.prompts import format_rag_messages
from contextforge.retrieval.embedder import BGEEmbedder
from contextforge.retrieval.faiss_store import FAISSStore
from contextforge.retrieval.retriever import DenseRetriever
from contextforge.storage.db import get_connection
from contextforge.storage.repositories import ChunkRepository

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT_VERSION = "rag_v1"
DEFAULT_CHUNK_PROFILE = "medium"
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_TOKENS = 1500
MAX_QUESTION_LENGTH = 4000


class RagPipeline:
    """Retrieval-Augmented Generation pipeline connecting dense retrieval to Qwen LLM."""

    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        model_loader: BaseModelLoader | None = None,
        generator: TextGenerator | None = None,
        formatter: ContextFormatter | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        chunk_profile: str = DEFAULT_CHUNK_PROFILE,
        db_path: Path | str | None = None,
        index_dir: Path | str | None = None,
        query_instruction_enabled: bool = False,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize RagPipeline.

        Args:
            retriever: Configured DenseRetriever instance (optional).
            model_loader: Configured BaseModelLoader instance (optional).
            generator: Configured TextGenerator instance (optional).
            formatter: Configured ContextFormatter instance (optional).
            top_k: Number of evidence chunks to retrieve.
            max_context_tokens: Maximum token budget for retrieved evidence context.
            prompt_version: Prompt template identifier (default: 'rag_v1').
            model_id: Model repository ID for generation.
            device: Compute device for generation ('auto', 'cuda', 'cpu').
            dtype: Weight precision for generation.
            max_new_tokens: Maximum tokens to generate.
            chunk_profile: Chunking profile index to query if retriever is created.
            db_path: Path to SQLite database if retriever is created.
            index_dir: Path to FAISS index directory if retriever is created.
            query_instruction_enabled: Whether to apply BGE retrieval query instruction.
            score_threshold: Provisional minimum retrieval score (optional).
        """
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.prompt_version = prompt_version
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.chunk_profile = chunk_profile
        self.score_threshold = score_threshold
        self._owns_db_conn = False
        self._db_conn = None

        # 1. Initialize retriever if not supplied
        if retriever is not None:
            self.retriever = retriever
        else:
            resolved_db = Path(db_path) if db_path else get_db_path()
            resolved_index = (
                Path(index_dir) if index_dir else Path("artifacts/faiss") / chunk_profile
            )

            if not resolved_index.is_dir():
                raise IndexNotFoundError(
                    f"FAISS index directory not found at '{resolved_index}'. "
                    f"Run 'python scripts/build_index.py --profile {chunk_profile}' first."
                )

            store = FAISSStore(chunk_profile=chunk_profile)
            store.load(resolved_index)

            embedder = BGEEmbedder()
            self._db_conn = get_connection(resolved_db)
            self._owns_db_conn = True
            chunk_repo = ChunkRepository(self._db_conn)

            self.retriever = DenseRetriever(
                embedder=embedder,
                store=store,
                chunk_repo=chunk_repo,
                top_k=top_k,
                query_instruction_enabled=query_instruction_enabled,
            )

        # 2. Initialize model loader if not supplied
        if model_loader is not None:
            self.model_loader = model_loader
        else:
            self.model_loader = BaseModelLoader(
                model_id=model_id,
                device=device,
                dtype=dtype,
            )

        # 3. Initialize generator if not supplied
        if generator is not None:
            self.generator = generator
        else:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            self.generator = TextGenerator(
                model_loader=self.model_loader,
                config=config,
            )

        # 4. Initialize context formatter if not supplied
        if formatter is not None:
            self.formatter = formatter
        else:
            tokenizer = (
                self.model_loader.tokenizer if hasattr(self.model_loader, "tokenizer") else None
            )
            self.formatter = ContextFormatter(
                max_context_tokens=max_context_tokens,
                tokenizer=tokenizer,
            )

    def close(self) -> None:
        """Close database connection if owned by this pipeline instance."""
        if self._owns_db_conn and self._db_conn:
            self._db_conn.close()

    def __enter__(self) -> RagPipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def validate_question(self, question: Any) -> str:
        """Validate input question string.

        Args:
            question: User input query.

        Returns:
            Cleaned question string.

        Raises:
            InvalidQuestionError: If input is not a string, is empty, or exceeds max length.
        """
        if not isinstance(question, str):
            raise InvalidQuestionError(f"Question must be a string, got {type(question).__name__}.")

        cleaned = question.strip()
        if not cleaned:
            raise InvalidQuestionError("Question cannot be empty or whitespace-only.")

        if len(cleaned) > MAX_QUESTION_LENGTH:
            raise InvalidQuestionError(
                f"Question exceeds maximum allowed length of {MAX_QUESTION_LENGTH} characters "
                f"(received {len(cleaned)} characters)."
            )

        return cleaned

    def generate(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Execute RAG question answering and return standardized AnswerResult.

        Flow:
            1. Validate input question
            2. DenseRetriever retrieves top-k chunks
            3. ContextFormatter constructs delimited evidence context adhering to token budget
            4. format_rag_messages constructs system and delimited user prompt
            5. TextGenerator executes greedy generation using Qwen model
            6. validate_citations checks citation integrity
            7. Return AnswerResult with pipeline='rag' and actual retrieved chunks in sources

        Args:
            question: Plain-text cybersecurity question.
            top_k: Optional override for number of chunks to retrieve.

        Returns:
            AnswerResult containing answer, actual retrieved sources, and telemetry metadata.
        """
        clean_question = self.validate_question(question)
        k = top_k if top_k is not None else self.top_k
        t_pipeline_start = time.perf_counter()

        # Step 1: Retrieval
        t_ret_0 = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(clean_question, top_k=k)
        retrieval_latency_ms = (time.perf_counter() - t_ret_0) * 1000.0

        # Optional provisional score filtering
        filtered_chunks = retrieved_chunks
        if self.score_threshold is not None:
            filtered_chunks = [c for c in retrieved_chunks if c.score >= self.score_threshold]

        # Step 2: Context formatting and token budgeting
        formatted_context: FormattedContext = self.formatter.format(filtered_chunks)

        # Step 3: Format prompt messages
        messages = format_rag_messages(
            question=clean_question,
            context_text=formatted_context.context_text,
            prompt_version=self.prompt_version,
        )

        # Step 4: Generation with shared Qwen generator
        t_gen_0 = time.perf_counter()
        gen_result = self.generator.generate(
            messages=messages,
            prompt_version=self.prompt_version,
        )
        generation_latency_ms = (time.perf_counter() - t_gen_0) * 1000.0

        total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        # Step 5: Validate citation integrity
        citation_validation: CitationValidationResult = validate_citations(
            answer=gen_result.answer,
            allowed_labels=formatted_context.label_to_chunk.keys(),
        )

        # Step 6: Construct metadata
        embedding_model = getattr(
            self.retriever.embedder,
            "model_name",
            "BAAI/bge-small-en-v1.5",
        )
        chunk_profile = getattr(
            self.retriever.store,
            "chunk_profile",
            self.chunk_profile,
        )

        metadata: dict[str, Any] = {
            "prompt_version": self.prompt_version,
            "model_id": self.model_loader.model_id,
            "embedding_model": embedding_model,
            "chunk_profile": chunk_profile,
            "top_k_requested": k,
            "retrieved_count": formatted_context.retrieved_count,
            "included_count": formatted_context.included_count,
            "context_token_count": formatted_context.context_token_count,
            "input_token_count": gen_result.input_token_count,
            "output_token_count": gen_result.output_token_count,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "citation_labels_used": citation_validation.used_labels,
            "invalid_citations": citation_validation.invalid_labels,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        }

        # Step 7: Return standardized AnswerResult
        # Note: sources is ALWAYS application-owned actual retrieved chunks
        return AnswerResult(
            question=clean_question,
            pipeline="rag",
            answer=gen_result.answer,
            model_id=self.model_loader.model_id,
            latency_ms=total_latency_ms,
            sources=retrieved_chunks,
            metadata=metadata,
        )

    def __call__(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Convenience callable alias for generate()."""
        return self.generate(question, top_k=top_k)
