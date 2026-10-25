# Project State

> **Rule:** The coding agent MUST read this file before making any change.
> Update this file at the end of every meaningful task.

---

## Current phase: 8 — COMPLETE

**Status:** COMPLETE

---

## Phase 8 acceptance criteria

| Criterion | Status |
|---|---|
| QLoRA pilot training script with strict GPU enforcement | VERIFIED |
| Training hyperparameter configuration (`config/finetune.yaml`) | VERIFIED |
| Safe preflight execution with token length verification | VERIFIED |
| QLoRA pilot training executed on NVIDIA GPU (GeForce RTX 2050, 4 GB VRAM) | VERIFIED |
| Zero CUDA OOM and clean convergence (train loss: 1.8394, val loss: 1.2016) | VERIFIED |
| PEFT adapter saved to artifacts/finetuning/phase8_pilot/ | VERIFIED |
| Adapter reload and single-inference smoke test verified | VERIFIED |
| Qualitative pilot smoke script executed with Base vs Fine-tuned comparison | VERIFIED |
| Zero leakage against Phase 7 test split (0/6) and Phase 6 evaluation split (0/6) | VERIFIED |
| Out-of-domain safety check passed (normal chocolate chip cookie response) | VERIFIED |
| Full unit regression suite passes (116 passed, 0 failed) | VERIFIED |
| Decision D-P08 approved in `docs/DECISIONS.md` | VERIFIED |
| Experiment EXP-005 recorded in `docs/EXPERIMENT_LOG.md` | VERIFIED |

---

## Phase 7 — Fine-Tuning Dataset Preparation

**Status:** COMPLETE

### Completed

- [x] Fine-tuning dataset builder implemented (`scripts/build_finetuning_dataset.py`):
  - 55 hand-crafted, grounded QA pairs across OWASP Top 10:2025 (27), NIST CSF 2.0 (15), and CISA CPG 2.0 (13).
  - Chat-style JSONL format: system, user, assistant messages compatible with Qwen2.5 chat template.
  - All answers grounded in approved corpus documents. No LLM-generated synthetic data.
  - Covers all 7 approved question categories: factual_definition (22), mitigation (9), control_guidance (8), recommendation (7), framework_mapping (4), procedural_guidance (4), risk_explanation (1).
- [x] Deterministic 70/15/15 split via SHA-256 hash sort: 38 train, 8 validation, 9 test.
- [x] Evaluation leakage prevention: all 12 `evaluation_v1.jsonl` questions verified absent from fine-tuning dataset.
- [x] Duplicate detection: zero internal question duplicates.
- [x] Fake citation check: zero `[SOURCE_N]` patterns in answers.
- [x] Source provenance validation: all 55 records map to valid `corpus_manifest.csv` document IDs.
- [x] Dataset outputs written to `data/qa/`:
  - `finetune_full.jsonl` (55 records, 65 KB)
  - `finetune_train.jsonl` (38 records)
  - `finetune_val.jsonl` (8 records)
  - `finetune_test.jsonl` (9 records)
  - `finetune_review_sample.jsonl` (18 balanced examples for human review)
  - `finetune_stats.json` (split statistics)
  - `finetune_metadata.json` (version, sources, format, split method)
- [x] Decision D-P07 recorded in `docs/DECISIONS.md`.
- [x] Tests: 21 Phase 7 unit tests + 99 total unit tests passing.

---

## Phase 8 — Pilot QLoRA Fine-Tuning

**Status:** COMPLETE

### Completed

- [x] Dedicated GPU environment configured (`.venv-gpu`) with PyTorch CUDA 13.0, Transformers 5.17.0, PEFT 0.21.2, TRL 1.14.1, bitsandbytes 0.50.2 on NVIDIA GeForce RTX 2050 (4 GB VRAM).
- [x] QLoRA training script with strict GPU enforcement (`scripts/train_qlora.py`).
- [x] Hyperparameter configuration defined (`config/finetune.yaml`).
- [x] Preflight dry-run verification passing (`--preflight`).
- [x] TRL 1.14.1 API compatibility translations verified (`warmup_ratio=0.1` -> `warmup_steps=2`, `max_length=512`, `processing_class=tokenizer`).
- [x] Physical pilot QLoRA training executed: 38 train records, 15/15 optimizer steps, train loss 1.8394, val loss 1.2016 in ~86.47s without CUDA OOM.
- [x] Adapter persisted to `artifacts/finetuning/phase8_pilot/` (`adapter_model.safetensors`, `adapter_config.json`, `training_metadata.json`).
- [x] PEFT adapter reload verified via `FineTunedPipeline`.
- [x] Leakage audit on qualitative smoke set: replaced leaking Q4 with fresh OWASP A10 question (0 test overlap, 0 Phase 6 eval overlap, max Jaccard 0.375).
- [x] Qualitative pilot smoke comparison executed (`scripts/run_pilot_smoke.py --device cuda`): Base vs Fine-tuned comparison across 6 questions; no generation regressions observed; normal out-of-domain cookie response.
- [x] Experiment EXP-005 recorded in `docs/EXPERIMENT_LOG.md`.
- [x] Decision D-P08 approved in `docs/DECISIONS.md`.
- [x] Unit test suite: 116 passed in 40.38s.

---

## Active task

Phase 8 complete — awaiting human approval to begin Phase 9.

---

## Blockers

None.

---

## Last verified commands

```text
.venv\Scripts\python.exe -m pytest tests/unit/ -v --tb=short
  -> 116 passed, 4 warnings in 40.38s

.venv-gpu\Scripts\python.exe scripts/train_qlora.py
  -> Completed 15/15 steps on NVIDIA RTX 2050; train loss: 1.8394, val loss: 1.2016

.venv-gpu\Scripts\python.exe scripts/run_pilot_smoke.py --device cuda
  -> Completed qualitative Base vs Fine-tuned smoke comparison across 6 questions
```

---

## Next allowed task

> **Phase 9 — Final Comparative Evaluation**
>
> Gate: Human confirms Phase 8 completion and approves start of Phase 9.
> Tasks: Compare Base, RAG, and Fine-Tuned pipelines across evaluation benchmark with formal comparative metrics.
> Do NOT begin Phase 9 without explicit approval.


---

## Phase 0 — Repository, Environment, and Guardrails

**Status:** COMPLETE (see git commit 55be644)

All Phase 0 acceptance criteria passed and verified. Full record in commit history.

---

## Phase 1 — Domain and Dataset Contract

**Status:** COMPLETE (see git commit ba43948)

All Phase 1 acceptance criteria passed and verified. Full record in commit history.

---

## Phase 2 — Document Ingestion

**Status:** COMPLETE

### Completed

- [x] Official NIST CSF 2.0 PDF downloaded from primary source (`https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf`), verified via PyMuPDF (32 pages, 68,335 chars), SHA-256 computed.
- [x] Official CISA CPG 2.0 PDF downloaded from primary source (`https://www.cisa.gov/sites/default/files/2025-12/CPG_Report_2.0_508c.pdf`), verified via PyMuPDF (36 pages, 95,324 chars), SHA-256 computed.
- [x] Canonical OWASP Top 10:2025 corpus acquired from official GitHub repository (`OWASP/Top10/2025/docs/en/*.md`, 16 Markdown files), individual SHA-256 hashes computed.
- [x] OWASP slide deck PDF discarded from canonical corpus role (retaining official Markdown documentation as canonical textual source).
- [x] Conservative text cleaner implemented (`src/contextforge/ingestion/cleaner.py`):
  - Preserves punctuation, casing, code blocks, URLs, and cybersecurity tokens (`CVE-2025-...`, `TLS 1.3`, `HTTP/2`, `XSS`, `SQL`, `< >`, `/`, `_`, `-`, `:`).
  - Normalizes via Unicode NFKC (handles ligatures like fi, fl).
  - Normalizes CRLF/CR to LF, strips non-printable ASCII control characters.
- [x] PDF loader implemented (`src/contextforge/ingestion/pdf_loader.py`):
  - PyMuPDF extraction with page-level provenance (1-indexed page numbers, char counts).
  - Scanned / image-only / empty PDF detection raising `NoExtractableTextError` (no silent OCR).
  - Safe file closure via `try...finally: doc.close()`.
- [x] Markdown loader implemented (`src/contextforge/ingestion/markdown_loader.py`):
  - Preserves headings and section-level provenance.
  - Full UTF-8 support.
- [x] Common types and error hierarchy implemented (`src/contextforge/common/types.py`):
  - `Document`, `ExtractedUnit`, `ExtractedDocument`.
  - `DocumentNotFoundError`, `UnsupportedDocumentTypeError`, `DocumentExtractionError`, `NoExtractableTextError`, `DuplicateDocumentError`.
- [x] SQLite storage layer implemented (`src/contextforge/storage/`):
  - `schema.sql`: `documents` table with `id`, `title`, `source_path`, `checksum` (UNIQUE), `source_type`, `status`, `created_at`, `metadata_json`.
  - `db.py`: Connection manager with WAL mode and foreign keys enabled.
  - `repositories.py`: Parameterized CRUD operations on `documents` table.
- [x] Ingestion service implemented (`src/contextforge/ingestion/service.py`):
  - End-to-end pipeline: path validation -> type detection -> SHA-256 -> duplicate check -> extraction -> conservative clean -> SQLite persistence.
  - Duplicate detection by SHA-256: returns `ALREADY_INGESTED` and skips insert (never creates duplicate rows).
- [x] Ingestion CLI implemented (`scripts/ingest.py`):
  - Supports single file and directory ingestion.
  - Structured output (`Status`, `Document ID`, `Title`, `Source type`, `SHA-256`, `Pages/sections`, `Characters`, `Database`).
  - Clean exit codes (0 for success/duplicate skipped, 1 for failure).
- [x] Real corpus ingested into `data/contextforge.db`:
  - NIST CSF 2.0 (1 PDF document, 32 pages)
  - CISA CPG 2.0 (1 PDF document, 36 pages)
  - OWASP Top 10:2025 (16 Markdown documents, 151 sections)
  - Total in DB: 18 documents.
- [x] Idempotency / duplicate prevention verified on real corpus (re-running ingestion outputs `ALREADY_INGESTED`, DB count remains 18).
- [x] Test suite: 32 tests passing (unit tests for cleaner, PDF loader, Markdown loader, hashing, storage, service, errors; integration test for CLI).
- [x] Manifest updated (`data/manifests/corpus_manifest.csv`) with 18 records, real dates, real paths, real SHA-256 hashes, zero fabricated data.
- [x] Architectural decisions recorded in `docs/DECISIONS.md` (D-011 through D-014).

---

## Phase 3 — Chunking, Embeddings, and FAISS

**Status:** COMPLETE

### Completed

- [x] Hugging Face `sentence-transformers==6.1.0`, `faiss-cpu==1.15.1`, and PyTorch installed and verified on Windows CPU (Python 3.13).
- [x] Model load verified for approved `BAAI/bge-small-en-v1.5` (dimension 384, L2 normalization confirmed).
- [x] RAG configuration created (`config/rag.yaml`) specifying embedding model, query instruction prefix, and 3 chunking profiles (`small`, `medium`, `large`).
- [x] Data contracts updated (`src/contextforge/common/types.py`):
  - `Chunk` with full provenance (`id`, `document_id`, `chunk_index`, `chunk_profile`, `text`, `token_count`, `content_hash`, `page_start`, `page_end`, `section_heading`, `metadata`).
  - `RetrievedChunk` with similarity `score`, 1-indexed `rank`, and citation `source_label`.
  - Exception hierarchy: `RetrievalError`, `IndexNotFoundError`, `IncompatibleIndexError`, `EmbeddingError`.
- [x] SQLite schema updated (`src/contextforge/storage/schema.sql` and `db.py`):
  - `chunks` table with `chunk_profile`, `section_heading`, `metadata_json`, and profile index.
  - Automatic migration implemented in `init_db` ensuring idempotency.
  - `ChunkRepository` implemented in `src/contextforge/storage/repositories.py` (batch save, lookup by id/ids, delete by profile/doc, count, list).
- [x] Token-aware chunker implemented (`src/contextforge/ingestion/chunker.py`):
  - Exact token counting via `BAAI/bge-small-en-v1.5` `AutoTokenizer` (never heuristic approximations).
  - Natural boundary segmentation (paragraphs, sentences) respecting document units.
  - Slicing for oversized sentences guaranteeing `token_count <= chunk_size_tokens`.
  - Provenance preservation: page numbers for PDF, headings for Markdown.
  - Deterministic chunk IDs: `{document_id}_{profile}_{chunk_index:04d}`.
  - Zero empty chunks guaranteed.
- [x] Dense embedder wrapper implemented (`src/contextforge/retrieval/embedder.py`):
  - `BGEEmbedder` wrapping `SentenceTransformer("BAAI/bge-small-en-v1.5")`.
  - Normalized 384-dimensional float32 embeddings.
  - Query instruction prefix support (`"Represent this sentence for searching relevant passages: "`).
  - Passages strictly never receive the query prefix.
- [x] FAISS vector store implemented (`src/contextforge/retrieval/faiss_store.py`):
  - `IndexFlatIP(384)` exact inner-product search (cosine similarity on normalized vectors).
  - Artifact persistence: `index.faiss`, `mapping.json`, `metadata.json`.
  - Metadata verification on load (dimension, model name, profile, vector count).
- [x] Dense retriever implemented (`src/contextforge/retrieval/retriever.py`):
  - Coordinates embedder, FAISS store, and SQLite chunk repository.
  - Formats human-readable citation source labels (e.g. `[nist-csf-20-P12]`, `[a01-2025-broken-access-control-How-to-Prevent]`).
- [x] CLI tools implemented:
  - `scripts/build_index.py`: builds SQLite chunks and FAISS indices for `--profile` or `--all`.
  - `scripts/search.py`: semantic search CLI returning ranked chunks with provenance and content previews.
  - `scripts/evaluate_retrieval.py`: automated evaluation harness measuring Recall@1/3/5, MRR, and query latency.
- [x] Grounded smoke question dataset created (`data/qa/retrieval_smoke.jsonl`): 15 questions across NIST CSF 2.0, CISA CPG 2.0, and OWASP Top 10:2025.
- [x] Controlled experiments executed and logged in `docs/EXPERIMENT_LOG.md`:
  - **EXP-001 (Chunking Profiles):** Built indices for small (638 chunks), medium (394 chunks), and large (281 chunks). All achieve 100% Recall@5.
  - **EXP-002 (Query Instruction Ablation):** Evaluated instruction vs raw query text across all profiles. Medium profile without generic instruction prefix achieved study-leading metrics: 86.7% Recall@1, 100% Recall@3 and Recall@5, 0.933 MRR, 16.8ms latency.
  - Selected `medium` profile (`chunk_size_tokens=300`, `overlap_tokens=50`) as production default in `config/rag.yaml`.
- [x] Decisions recorded in `docs/DECISIONS.md`: D-P03 (Embedding model approved), D-P05 (Chunking parameters approved), D-015 (FAISS Flat IP), D-016 (Query instruction prefix).
- [x] Test suite: 49 tests passing with zero warnings (unit tests for chunker, embedder, FAISS store, retriever, cleaner, loaders, storage; integration tests for search CLI and end-to-end retrieval).

---

## Active task

Phase 3 complete — awaiting human approval to begin Phase 4.

---

## Blockers

None.

---

## Last verified commands

```text
.venv\Scripts\python.exe scripts/build_index.py --all
  -> Built 'small' (638 chunks, 27.01s), 'medium' (394 chunks, 29.15s), 'large' (281 chunks, 32.71s)

.venv\Scripts\python.exe scripts/evaluate_retrieval.py --all-profiles
  -> small: Recall@1=0.733, Recall@3=1.000, Recall@5=1.000, MRR=0.844, Latency=20.9ms
  -> medium: Recall@1=0.733, Recall@3=0.867, Recall@5=1.000, MRR=0.833, Latency=19.2ms
  -> large: Recall@1=0.733, Recall@3=0.800, Recall@5=1.000, MRR=0.813, Latency=22.7ms

.venv\Scripts\python.exe scripts/evaluate_retrieval.py --profile medium --no-instruction
  -> medium: Recall@1=0.867, Recall@3=1.000, Recall@5=1.000, MRR=0.933, Latency=16.8ms

.venv\Scripts\python.exe scripts/search.py "How does OWASP recommend preventing Broken Access Control vulnerabilities?" -k 3
  -> Rank 1: [a01-2025-broken-access-control-References] (score=0.8169)

.venv\Scripts\python.exe scripts/search.py "What is the Govern function in NIST CSF 2.0?" -k 2
  -> Rank 1: [cisa-cpg-20-P7-P8] (score=0.7833), Rank 2: [nist-csf-20-P8-P9] (score=0.7632)

.venv\Scripts\python.exe -m pytest -v
  -> 49 passed, 0 warnings in 61.22s
```

---

## Known issues / limitations

- Embeddings and FAISS search run on local CPU (Python 3.13); GPU acceleration not required for current corpus scale.
- Memory usage for FAISS index + BGE model is <1 GB RAM.

---

## Phase 3 acceptance criteria

| Criterion | Status |
|---|---|
| Chunks bounded by token count via BGE tokenizer | VERIFIED |
| No empty chunks generated across any profile | VERIFIED |
| Full provenance preserved (page ranges for PDF, headings for Markdown) | VERIFIED |
| Chunks persisted in SQLite with parameterized queries | VERIFIED |
| Embeddings generated using BAAI/bge-small-en-v1.5 (384 dimensions) | VERIFIED |
| Embeddings L2-normalized; cosine similarity via FAISS IndexFlatIP | VERIFIED |
| FAISS index, mapping, and metadata persisted to disk | VERIFIED |
| Deterministic rebuild command functional (`scripts/build_index.py`) | VERIFIED |
| Semantic search CLI functional (`scripts/search.py`) | VERIFIED |
| Controlled A/B/C chunking experiment executed and logged (EXP-001) | VERIFIED |
| Query instruction ablation executed and logged (EXP-002) | VERIFIED |
| Smoke QA dataset created with grounded questions (`retrieval_smoke.jsonl`) | VERIFIED |
| Retrieval evaluation script measures Recall@1/3/5, MRR, and latency | VERIFIED |
| Full test suite passes (49/49) | VERIFIED |
| Zero Phase 4 components introduced (no LLM inference, no answer generation) | VERIFIED |

---

## Phase 4 — Base LLM Pipeline

**Status:** COMPLETE

### Completed

- [x] Model decision approved and recorded in `docs/DECISIONS.md`: `Qwen/Qwen2.5-1.5B-Instruct` as single base causal language model (Apache-2.0, ~1.54B parameters).
- [x] Configuration created (`config/model.yaml`): deterministic generation (`do_sample: false`, temperature/top_p omitted), max_new_tokens=256, prompt=base_v1.
- [x] Common contracts updated (`src/contextforge/common/types.py`):
  - `AnswerResult` (`question`, `pipeline`, `answer`, `model_id`, `latency_ms`, `sources`, `metadata`).
  - Error hierarchy: `LLMError`, `ModelLoadError`, `GenerationError`, `InvalidQuestionError`.
- [x] Centralized, versioned prompt template created (`src/contextforge/llm/prompts.py`):
  - `base_v1`: concise defensive cybersecurity persona.
  - Zero retrieval placeholders; no false claims of external document access.
  - Structured chat template message formatting (`system` + `user`).
- [x] Model loader implemented (`src/contextforge/llm/base_model.py`):
  - `BaseModelLoader` with automatic device selection (CUDA or CPU) and dtype resolution (`torch.bfloat16` on CPU).
  - Puts model in eval mode, loads once per process, exposes runtime diagnostic metadata.
- [x] Text generation wrapper implemented (`src/contextforge/llm/generation.py`):
  - `TextGenerator` using official Qwen chat template.
  - Strips input prompt tokens and decodes generated output tokens only.
  - Enforces deterministic greedy decoding (`do_sample=False`).
  - Captures exact input and output token counts.
- [x] Application pipeline implemented (`src/contextforge/pipelines/base.py`):
  - `BasePipeline` returning standard `AnswerResult` with `pipeline="base"` and `sources=[]`.
  - Input validation rejecting empty, whitespace-only, or oversized questions with explicit errors.
  - Measures wall-clock generation latency excluding cold model load time.
  - Zero FAISS, retrieval, or SQLite chunk dependencies.
- [x] CLI inference runner implemented (`scripts/run_base.py`):
  - Structured output displaying pipeline, model, device, latency, token counts, and generated answer.
- [x] Representative smoke questions executed (`scripts/evaluate_base_smoke.py`):
  1. What is broken access control? (Latency: 32.0s, In: 54 tok, Out: 110 tok)
  2. What is the purpose of multi-factor authentication? (Latency: 56.2s, In: 57 tok, Out: 256 tok)
  3. What is the Govern function in a cybersecurity framework? (Latency: 49.6s, In: 58 tok, Out: 195 tok)
  4. How can SQL injection risk be reduced? (Latency: 70.0s, In: 56 tok, Out: 256 tok)
  5. What is the difference between authentication and authorization? (Latency: 38.9s, In: 57 tok, Out: 103 tok)
- [x] Baseline limitation demonstrated:
  - Question: "What are the official ten vulnerability categories defined in OWASP Top 10:2025?"
  - Result: Base model hallucinates/recalls outdated 2017/2021 categories (XXE, Sensitive Data Exposure), missing 2025 additions (`Software Supply Chain Failures`, `Mishandling of Exceptional Conditions`). Concretely proves why RAG is necessary for accurate, current domain QA.
- [x] Tests:
  - Unit tests: 59 passed with mocked model/tokenizer (runs fast, no heavy downloads in ordinary test pass).
  - Integration test: `test_real_qwen_base_pipeline_inference` passed (verifies real Qwen2.5 loading, chat template, and generation).
  - Total test suite: 60/60 passed.

---

---

## Phase 5 — RAG Pipeline

**Status:** COMPLETE

### Completed

- [x] Configuration updated (`config/rag.yaml`) with `rag:` section (`top_k: 5`, `max_context_tokens: 1500`, `prompt_version: rag_v1`, `query_instruction_enabled: false`).
- [x] Versioned prompt template created (`src/contextforge/llm/prompts.py`):
  - `RAG_PROMPT_V1`: concise defensive cybersecurity persona, prompt injection defense clause, `<BEGIN_EVIDENCE>` / `<END_EVIDENCE>` boundary tags, strict refusal instruction when evidence is insufficient, and standardized source citation labels.
  - Helpers: `get_rag_system_prompt()`, `format_rag_messages()`, and prompt registry entry for `rag_v1`.
- [x] Context formatter implemented (`src/contextforge/llm/context_formatter.py`):
  - `ContextFormatter` mapping chunks to `[SOURCE_1]`, `[SOURCE_2]`... labels.
  - Enforces strict 1500-token budget (drops overflowing chunks without mid-chunk truncation).
  - Resolves human-readable document titles and section/page references (`resolve_document_source_name`, `resolve_reference_label`).
- [x] Citation validator implemented (`src/contextforge/llm/citations.py`):
  - `validate_citations()` and `extract_citation_labels()` to verify cited labels against retrieved chunks.
  - Detects and reports invalid/hallucinated citation labels in `metadata["invalid_citations"]`.
- [x] TextGenerator updated (`src/contextforge/llm/generation.py`):
  - Accepts optional pre-built `messages=` list for structured chat template while maintaining full backwards compatibility for BasePipeline `question=` calls.
- [x] RAG pipeline implemented (`src/contextforge/pipelines/rag.py`):
  - `RagPipeline` orchestrating `DenseRetriever` (medium profile, BGE embeddings, FAISS), `ContextFormatter`, `TextGenerator` (greedy deterministic Qwen2.5-1.5B-Instruct), and `validate_citations`.
  - Returns standardized `AnswerResult` with `pipeline="rag"`, populated `sources`, and detailed telemetry (`retrieval_latency_ms`, `generation_latency_ms`, `citation_labels_used`, `invalid_citations`, etc.).
- [x] BasePipeline isolation maintained (`src/contextforge/pipelines/base.py`):
  - Zero retrieval dependencies.
  - Regression verified by unit test `test_base_pipeline_isolation_regression`.
- [x] CLI inference runners implemented:
  - `scripts/run_rag.py`: Single-question RAG inference with citation breakdown and metadata display.
  - `scripts/compare_base_rag.py`: Side-by-side Base vs RAG comparison script.
- [x] Minimal Top-K comparison executed (EXP-003: K=3, 5, 8):
  - Evaluated retrieval and context metrics across domain queries.
  - Confirmed `top_k = 5` as moderate default balancing context coverage and token budget.
- [x] Real RAG qualitative verification executed on live Qwen model:
  - OWASP Top 10:2025 taxonomy query: Grounded answer citing official categories from retrieved evidence.
  - NIST CSF 2.0 Govern function query: Accurately explains Govern function citing NIST sources.
  - CISA CPG 2.0 query: Accurately explains CISA goals citing CISA source evidence.
  - Unsupported domain query ("best recipe for chocolate cake"): Model correctly refuses and states insufficient evidence based on prompt constraints.
- [x] Tests:
  - Unit tests: 70 passed (includes 15 Phase 5 unit tests for context formatter, citations, and rag pipeline).
  - Integration test: `test_real_rag_pipeline_end_to_end_inference` passed in live environment (83.82s).

---

## Phase 6 — Evaluation Harness v1: Base vs RAG

**Status:** COMPLETE

### Completed

- [x] Evaluation dataset created (`data/qa/evaluation_v1.jsonl`): 12 curated questions across OWASP Top 10:2025 (4), NIST CSF 2.0 (3), CISA CPG 2.0 (3), and unsupported out-of-corpus queries (2), with factual reference points and expected source categories.
- [x] Evaluation harness implemented (`src/contextforge/evaluation/`):
  - `EvaluationQuestion` schema and `load_evaluation_dataset` parser.
  - Deterministic `calculate_reference_coverage` using rule-based factual criteria verification.
  - `is_expected_source_retrieved` checking retrieved chunks against expected source authorities.
  - `is_unsupported_refusal` detecting proper refusal on queries outside the corpus.
  - `evaluate_question_pair` comparing Base and RAG answers on identical questions.
  - `aggregate_evaluation_results` computing macro-averages for coverage, latencies, tokens, citation validity, and refusal rates.
- [x] No LLM-as-a-judge dependency: evaluation is completely deterministic, transparent, and reproducible without external APIs.
- [x] Single-process shared model loader: Qwen2.5-1.5B-Instruct loaded once into memory to avoid repeated CPU cold starts.
- [x] Evaluation runner implemented (`scripts/evaluate_base_vs_rag.py`): executes both pipelines on all 12 questions under identical greedy deterministic decoding.
- [x] Raw results saved to `artifacts/evaluation/base_vs_rag_v1.json` (gitignored).
- [x] Unit test suite: 8 dedicated evaluation unit tests (`tests/unit/test_evaluation.py`) verifying dataset loading, metric calculations, expected-source matching, refusal checks, and serialization (78 total unit tests passing).
- [x] Experiment EXP-004 recorded in `docs/EXPERIMENT_LOG.md`.
