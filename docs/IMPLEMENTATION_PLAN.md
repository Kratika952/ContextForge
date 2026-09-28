# DomainAdapt — Complete Implementation Plan

> **Purpose:** This document is the primary technical source of truth for building DomainAdapt in small, verifiable phases.
>
> **Audience:** Human developer + AI coding agent.
>
> **Rule:** The coding agent must read this document and `docs/PROJECT_STATE.md` before making any project change. It must work on only the currently active phase unless the human explicitly changes scope.

---

## 0. Project Summary

### 0.1 Product

**DomainAdapt** is a domain-specific question-answering system that exposes three comparable inference paths:

1. **Base LLM** — the untouched pretrained/instruction model.
2. **RAG** — the same base LLM augmented with retrieved evidence from a domain document corpus.
3. **Fine-tuned LLM** — the same base LLM with a PEFT LoRA/QLoRA adapter trained on domain-specific instruction/QA examples.

The application lets a user ask the same question using any of the three paths, records results, and provides a dashboard for comparing quality and system performance.

### 0.2 Research Question

> **For domain-specific question answering, how do a base pretrained LLM, retrieval-augmented generation, and parameter-efficient fine-tuning compare in answer quality, groundedness/hallucination behavior, retrieval quality, latency, and resource requirements?**

### 0.3 Real-World Use Case

The final system should be usable as a **domain knowledge assistant** for a bounded document collection such as software documentation, cybersecurity guidance, university regulations, research literature, or another approved domain.

The research comparison is part of the project, but the application must still behave like a useful product.

---

# 1. Locked Technology Stack

Do **not** replace technologies in this table without explicit human approval.

| Layer | Technology |
|---|---|
| Language | Python |
| ML / DL | PyTorch |
| LLM runtime | Hugging Face Transformers |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Fine-tuning | PEFT + TRL + LoRA/QLoRA |
| Quantization | bitsandbytes where supported |
| Dataset utilities | Hugging Face Datasets + Pandas |
| PDF extraction | PyMuPDF |
| Backend API | FastAPI |
| App database | SQLite |
| Frontend | Streamlit |
| Metrics | scikit-learn + custom evaluation code |
| Visualizations | Matplotlib |
| Experiment environment | Local machine + Kaggle/Google Colab when GPU is needed |
| Version control | Git + GitHub |
| Testing | pytest |
| Configuration | environment variables + checked-in YAML/JSON config |
| Serialization | JSON / JSONL / CSV as appropriate |

### 1.1 Explicitly Out of Scope Unless Approved

Do not add these merely because they are popular:

- LangChain
- LlamaIndex
- PostgreSQL
- MongoDB
- Redis
- Celery
- Kafka
- Kubernetes
- Docker as a requirement
- AWS/GCP/Azure managed services
- paid OpenAI/Anthropic/Gemini APIs
- authentication/SSO
- multi-tenant architecture
- agent frameworks
- graph databases
- microservices

They may be considered later only after the complete baseline project works and a concrete need is demonstrated.

---

# 2. Decisions That Must Be Made Explicitly

The agent must **not silently choose** any of the following.

| Decision | Status | Where to record it |
|---|---|---|
| Domain | HUMAN_DECISION_REQUIRED | `docs/DECISIONS.md` |
| Corpus source(s) | HUMAN_DECISION_REQUIRED | `docs/DECISIONS.md` |
| Base LLM | PHASE_4_DECISION | `docs/DECISIONS.md` |
| Embedding model | PHASE_3_DECISION | `docs/DECISIONS.md` |
| Chunking parameters | PHASE_3_EXPERIMENT | config + experiment record |
| Top-k retrieval | PHASE_5_EXPERIMENT | config + experiment record |
| Fine-tuning dataset format | PHASE_7_DECISION | `docs/DECISIONS.md` |
| QLoRA hyperparameters | PHASE_8_EXPERIMENT | training config |
| Final evaluation thresholds | PHASE_9_DECISION | evaluation config |

If a required decision has not been made, stop at the decision gate and ask the human. Do not fabricate one.

---

# 3. Core Architecture

```text
                                DOMAIN DOCUMENTS
                                      |
                                      v
                         +---------------------------+
                         |  INGESTION / PREPROCESS   |
                         | PyMuPDF -> clean -> chunk |
                         +-------------+-------------+
                                       |
                      +----------------+----------------+
                      |                                 |
                      v                                 v
             +------------------+              +------------------+
             | SQLite metadata  |              | Embedding model  |
             | docs + chunks    |              | SentenceTransf.  |
             +------------------+              +--------+---------+
                                                        |
                                                        v
                                                  +-----------+
                                                  |   FAISS   |
                                                  | vector DB |
                                                  +-----+-----+
                                                        |
                                                        |
       +------------------------+-----------------------+------------------------+
       |                        |                                                |
       v                        v                                                v
+--------------+        +---------------+                              +------------------+
|   Base LLM   |        |  RAG Engine   |                              | Fine-tuned LLM   |
| same model   |        | retrieve +    |                              | Base + LoRA      |
| no corpus    |        | prompt + LLM  |                              | adapter          |
+------+-------+        +-------+-------+                              +---------+--------+
       |                        |                                                |
       +------------------------+----------------------+-------------------------+
                                                       |
                                                       v
                                              +-----------------+
                                              | FastAPI Backend |
                                              +--------+--------+
                                                       |
                              +------------------------+-----------------------+
                              |                                                |
                              v                                                v
                     +----------------+                                +------------------+
                     | SQLite results |                                | Streamlit UI     |
                     | queries/runs   |                                | ask + compare    |
                     +-------+--------+                                +---------+--------+
                             |                                                   |
                             +----------------------+----------------------------+
                                                    |
                                                    v
                                           +-------------------+
                                           | Evaluation Engine |
                                           | offline benchmark |
                                           +---------+---------+
                                                     |
                                                     v
                                           +-------------------+
                                           | Reports / graphs  |
                                           +-------------------+
```

---

# 4. Architectural Principles

## 4.1 Same Base Model for Fair Comparison

The base and RAG paths must use the **same exact base model**.

The fine-tuned path must use:

```text
same base model + LoRA/QLoRA adapter
```

Do not compare unrelated models and call the result “RAG vs fine-tuning.”

## 4.2 Retrieval and Metadata Are Separate

FAISS stores/searches vectors. SQLite is the source for structured document/chunk metadata.

Recommended mapping:

```text
FAISS vector id -> chunk_id -> SQLite chunks table -> document metadata
```

Do not use FAISS as the application database.

## 4.3 Evaluation Is Offline-First

The evaluation framework is primarily a reproducible CLI/script workflow, not a web endpoint.

The frontend may visualize saved results, but evaluation logic must not depend on Streamlit.

## 4.4 Core ML Logic Must Not Depend on the UI

Streamlit must call FastAPI.

Streamlit must not contain model-loading, FAISS indexing, fine-tuning, or evaluation logic.

## 4.5 Reproducibility Over Cleverness

Every experiment must record:

- timestamp
- Git commit if available
- model identifier
- embedding model identifier
- config
- random seed
- dataset version/hash
- metrics
- environment notes

## 4.6 No Test Leakage

Test questions and answers must never be used to:

- generate training examples
- select hyperparameters
- tune prompts
- choose chunk size
- choose top-k
- decide fine-tuning epochs

A validation set is used for those decisions.

---

# 5. Repository Structure

Create this structure gradually. Do not create empty complexity that is not needed by the active phase.

```text
DomainAdapt/
|
|-- README.md
|-- pyproject.toml or requirements.txt
|-- .gitignore
|-- .env.example
|
|-- docs/
|   |-- IMPLEMENTATION_PLAN.md
|   |-- PROJECT_STATE.md
|   |-- DECISIONS.md
|   |-- EXPERIMENT_LOG.md
|   `-- architecture/
|
|-- config/
|   |-- project.yaml
|   |-- rag.yaml
|   |-- model.yaml
|   `-- evaluation.yaml
|
|-- data/
|   |-- raw/                 # ignored if large/private
|   |-- processed/
|   |-- qa/
|   |   |-- train.jsonl
|   |   |-- validation.jsonl
|   |   `-- test.jsonl
|   `-- manifests/
|
|-- artifacts/
|   |-- faiss/
|   |-- adapters/
|   |-- evaluation/
|   `-- plots/
|
|-- src/
|   `-- domainadapt/
|       |-- __init__.py
|       |
|       |-- common/
|       |   |-- config.py
|       |   |-- logging.py
|       |   `-- types.py
|       |
|       |-- ingestion/
|       |   |-- pdf_loader.py
|       |   |-- cleaner.py
|       |   |-- chunker.py
|       |   `-- service.py
|       |
|       |-- storage/
|       |   |-- db.py
|       |   |-- repositories.py
|       |   `-- schema.sql
|       |
|       |-- retrieval/
|       |   |-- embedder.py
|       |   |-- faiss_store.py
|       |   `-- retriever.py
|       |
|       |-- llm/
|       |   |-- base_model.py
|       |   |-- prompts.py
|       |   `-- generation.py
|       |
|       |-- pipelines/
|       |   |-- base.py
|       |   |-- rag.py
|       |   `-- finetuned.py
|       |
|       |-- finetuning/
|       |   |-- dataset.py
|       |   |-- train.py
|       |   `-- adapter.py
|       |
|       |-- evaluation/
|       |   |-- dataset.py
|       |   |-- retrieval_metrics.py
|       |   |-- answer_metrics.py
|       |   |-- runner.py
|       |   `-- report.py
|       |
|       `-- api/
|           |-- main.py
|           |-- schemas.py
|           |-- dependencies.py
|           `-- routes/
|
|-- frontend/
|   `-- streamlit_app.py
|
|-- scripts/
|   |-- ingest.py
|   |-- build_index.py
|   |-- run_base.py
|   |-- run_rag.py
|   |-- run_finetuned.py
|   |-- train_adapter.py
|   `-- evaluate.py
|
`-- tests/
    |-- unit/
    |-- integration/
    `-- fixtures/
```

---

# 6. Shared Data Contracts

Freeze these concepts early so separate pipelines remain comparable.

## 6.1 Document

Minimum fields:

```python
Document:
    id: str
    title: str
    source_path: str
    checksum: str
    created_at: datetime
    metadata: dict
```

## 6.2 Chunk

```python
Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int
    content_hash: str
```

## 6.3 RetrievedChunk

```python
RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    source_label: str
```

## 6.4 AnswerResult

Every inference path returns the same top-level shape:

```python
AnswerResult:
    question: str
    pipeline: Literal["base", "rag", "finetuned"]
    answer: str
    model_id: str
    latency_ms: float
    sources: list[RetrievedChunk]
    metadata: dict
```

For base and fine-tuned modes, `sources` is normally empty.

Do not let each pipeline invent incompatible result structures.

---

# 7. SQLite Data Model

Start minimal and migrate carefully.

## 7.1 `documents`

```text
id                  TEXT PRIMARY KEY
title               TEXT
source_path          TEXT
checksum             TEXT UNIQUE
status               TEXT
created_at           TEXT
metadata_json        TEXT
```

## 7.2 `chunks`

```text
id                  TEXT PRIMARY KEY
document_id         TEXT NOT NULL
chunk_index         INTEGER NOT NULL
text                TEXT NOT NULL
page_start          INTEGER
page_end            INTEGER
token_count         INTEGER
content_hash        TEXT
embedding_model     TEXT
FOREIGN KEY(document_id) REFERENCES documents(id)
```

## 7.3 `queries`

```text
id                  TEXT PRIMARY KEY
question            TEXT NOT NULL
created_at           TEXT
```

## 7.4 `responses`

```text
id                  TEXT PRIMARY KEY
query_id            TEXT NOT NULL
pipeline            TEXT NOT NULL
answer              TEXT NOT NULL
model_id            TEXT NOT NULL
latency_ms          REAL
prompt_version      TEXT
created_at           TEXT
metadata_json       TEXT
FOREIGN KEY(query_id) REFERENCES queries(id)
```

## 7.5 `retrieval_events`

```text
id                  TEXT PRIMARY KEY
response_id         TEXT NOT NULL
chunk_id            TEXT NOT NULL
rank                INTEGER
score               REAL
FOREIGN KEY(response_id) REFERENCES responses(id)
FOREIGN KEY(chunk_id) REFERENCES chunks(id)
```

## 7.6 `evaluation_runs`

```text
id                  TEXT PRIMARY KEY
name                TEXT
config_json         TEXT
dataset_hash        TEXT
git_commit          TEXT
created_at           TEXT
summary_json        TEXT
```

Do not add user/auth tables in the initial project.

---

# 8. API Contract

Initial API:

## `GET /health`

Returns application health and which components are loaded.

## `POST /ask`

Request:

```json
{
  "question": "How is SQL injection prevented?",
  "pipeline": "rag"
}
```

Response conceptually matches `AnswerResult`.

## `POST /documents`

Upload/import a supported document. In early phases this may remain a CLI-only operation; add API upload only after ingestion is stable.

## `GET /responses/{id}`

Returns a stored response and sources.

## `GET /comparison`

Returns saved aggregate experiment/evaluation summaries for the UI.

### Not Required in MVP

- `/login`
- `/register`
- `/agents`
- `/chat/history` with multi-turn memory
- live fine-tuning endpoint
- live reindex-on-every-request
- background job infrastructure

---

# 9. Prompt Versioning

Prompts are code/config, not random strings scattered around the repository.

Example:

```text
RAG_PROMPT_V1
SYSTEM:
You answer only from the provided evidence.
If evidence is insufficient, say that the supplied documents do not contain enough information.
Cite the provided source labels for factual claims.

CONTEXT:
{context}

QUESTION:
{question}
```

Every saved response should record a `prompt_version`.

Changing a prompt during evaluation creates a new experiment version.

---

# 10. Phase-by-Phase Implementation

No phase is complete until its acceptance criteria pass.

---

## PHASE 0 — Repository, Environment, and Guardrails

### Goal

Create a reproducible project skeleton with no ML functionality yet.

### Tasks

1. Initialize Git repository.
2. Add core directories.
3. Add this file at `docs/IMPLEMENTATION_PLAN.md`.
4. Add `docs/PROJECT_STATE.md`.
5. Add `docs/DECISIONS.md`.
6. Add `docs/EXPERIMENT_LOG.md`.
7. Create Python virtual environment instructions.
8. Add dependency file.
9. Add `.env.example`.
10. Add `.gitignore` for:
    - virtual environments
    - model caches where appropriate
    - raw private data
    - generated FAISS indexes
    - adapters too large for Git
    - SQLite runtime DB if desired
    - secrets
11. Configure pytest.
12. Add a trivial import/health smoke test.
13. Create `config/project.yaml`.

### `PROJECT_STATE.md` Initial Shape

```markdown
# Project State

Current phase: 0
Status: IN_PROGRESS

## Completed
- [ ]

## Active task
...

## Blockers
None

## Last verified commands
...

## Next allowed task
...
```

### Acceptance Criteria

- `python -c "import domainadapt"` succeeds.
- `pytest` succeeds.
- No secrets are committed.
- The implementation plan is present.
- Project state identifies Phase 0.
- No LangChain/LlamaIndex/database substitution has appeared.

### Forbidden in This Phase

- model downloads
- RAG implementation
- fine-tuning
- Streamlit UI
- premature API routes

---

## PHASE 1 — Domain and Dataset Contract

### Goal

Define what the system knows and what counts as a correct answer.

### Human Decision Gate

The human chooses the domain and approves corpus sources.

Recommended beginner-friendly default: **software/developer documentation**.

### Tasks

1. Record domain in `docs/DECISIONS.md`.
2. Define allowed document types.
3. Define data licensing/source notes.
4. Create a corpus manifest:
   - source
   - title
   - file
   - version/date if known
   - checksum
5. Define inclusion/exclusion rules.
6. Define the QA task:
   - factual QA?
   - procedural QA?
   - both?
7. Define what an answer must contain.
8. Define test-set construction rules.
9. Define leakage prevention.
10. Put a small initial corpus under `data/raw/`.

### Acceptance Criteria

- Domain is explicit.
- Corpus source list is explicit.
- At least 3 representative documents are available for pipeline development.
- No test answers have been used for model tuning.
- Corpus manifest exists.

### Exit Deliverable

`data/manifests/corpus_manifest.*`

---

## PHASE 2 — Document Ingestion

### Goal

Reliably turn supported documents into normalized text with provenance.

### Tasks

1. Implement PDF extraction with PyMuPDF.
2. Preserve:
   - filename
   - document title when known
   - page numbers
   - extraction warnings
3. Implement conservative text cleaning.
4. Do **not** aggressively remove punctuation or case.
5. Detect empty/scanned PDFs and return a clear unsupported/OCR-needed message.
6. Compute document checksum.
7. Prevent duplicate ingestion using checksum.
8. Insert documents into SQLite.
9. Add tests using tiny fixture PDFs/text fixtures.
10. Add CLI:

```bash
python scripts/ingest.py <path>
```

### Acceptance Tests

- Text can be extracted from a normal text PDF.
- Page provenance survives extraction.
- Duplicate file is detected.
- Missing/corrupt file fails with a clear error.
- No OCR dependency is silently introduced.
- Unit tests pass.

### Exit Deliverable

A deterministic normalized document representation.

---

## PHASE 3 — Chunking, Embeddings, and FAISS

### Goal

Create the searchable retrieval corpus.

### Decision Gate

Select and record the embedding model after a small compatibility test.

### Tasks

1. Implement one simple chunking strategy first.
2. Chunk by token/character window with overlap.
3. Preserve chunk provenance.
4. Store chunk rows in SQLite.
5. Compute embeddings with Sentence Transformers.
6. Build FAISS index.
7. Persist:
   - FAISS index
   - vector-id-to-chunk-id mapping
   - embedding model identifier
   - index build metadata
8. Create deterministic rebuild command.
9. Implement semantic search.
10. Add `top_k` to config.
11. Add tests:
    - expected number of chunks
    - no empty chunks
    - search returns requested K
    - returned IDs resolve to SQLite chunks

### Initial Experiment

Try only a small controlled set of chunk configurations, e.g.:

```text
A: smaller chunks + overlap
B: medium chunks + overlap
C: larger chunks + overlap
```

Do not generate dozens of variants.

### Acceptance Criteria

For a hand-written smoke set of questions, obviously relevant passages appear in the top results often enough to justify moving forward.

Formal retrieval evaluation comes later.

### Exit Deliverables

- persisted FAISS index
- persisted mapping
- chunk metadata in SQLite
- retriever API

---

## PHASE 4 — Base LLM Pipeline

### Goal

Create the untouched baseline inference system.

### Decision Gate

Choose a small open instruct model that:

- fits available hardware for inference
- is compatible with Transformers
- has a license suitable for the project
- can later be adapted with LoRA/QLoRA
- provides stable text generation

Do not pick a huge model merely for prestige.

### Tasks

1. Implement base model loader.
2. Centralize generation config:
   - max new tokens
   - temperature
   - top-p if used
   - seed where meaningful
3. Implement base prompt.
4. Implement `BasePipeline`.
5. Return `AnswerResult`.
6. Measure wall-clock latency.
7. Add CLI smoke test.
8. Add tests with model calls mocked at unit-test level.
9. Add one manual real-model integration smoke test.

### Acceptance Criteria

- Same question produces a valid `AnswerResult`.
- Model identifier is recorded.
- Latency is recorded.
- Pipeline does not access FAISS.
- No LoRA adapter is loaded.
- Output failures are handled cleanly.

---

## PHASE 5 — RAG Pipeline

### Goal

Answer using retrieved domain evidence.

### Flow

```text
question
 -> embedding
 -> FAISS top-k
 -> chunk metadata from SQLite
 -> context formatting
 -> RAG prompt
 -> SAME base LLM
 -> answer + sources
```

### Tasks

1. Implement `Retriever`.
2. Implement context formatter.
3. Add source labels such as `[DOC1-P12]`.
4. Implement versioned RAG prompt.
5. Implement `RagPipeline`.
6. Ensure same base LLM as Phase 4.
7. Return sources in `AnswerResult`.
8. Handle “insufficient evidence” behavior.
9. Add integration tests.
10. Record retrieval ranks/scores.

### Guardrails

- Do not dump unlimited chunks into the prompt.
- Do not claim citations that were not retrieved.
- Do not let the LLM invent source filenames.
- Source labels are generated by application code.

### Acceptance Criteria

- RAG answers include retrievable source metadata.
- Returned sources correspond to actual chunks.
- Base and RAG paths use the same base model ID.
- Missing evidence produces an explicit insufficient-evidence response rather than fabricated certainty.
- Tests pass.

---

## PHASE 6 — Evaluation Harness v1: Base vs RAG

### Goal

Build the evaluation infrastructure **before fine-tuning**.

### Test Dataset Schema

Recommended JSONL fields:

```json
{
  "id": "q001",
  "question": "...",
  "reference_answer": "...",
  "relevant_document_ids": ["..."],
  "relevant_chunk_ids": ["..."],
  "category": "factual"
}
```

Chunk IDs may be omitted initially if manually expensive, but document-level relevance should exist when possible.

### Metrics

#### Retrieval

- Recall@1
- Recall@3
- Recall@5
- MRR where relevant

#### Answer

Use a mix of:

- token/word F1 where appropriate
- semantic similarity for paraphrased answers
- task-specific exact matching for deterministic facts
- human rubric on a representative sample

#### Groundedness / Hallucination

Do **not** claim that one automatic score fully measures hallucination.

Use:

1. citation/evidence support checks
2. optional NLI-based support signal
3. human review subset

If an LLM-as-judge is ever used, label it clearly as an automated judge and keep it separate from human evaluation.

#### System

- latency
- peak GPU/CPU memory where practical
- index size
- model/adaptor size

### Tasks

1. Create validation and test schemas.
2. Add evaluation runner.
3. Save raw predictions before computing summary metrics.
4. Save experiment config.
5. Generate CSV/JSON report.
6. Generate basic Matplotlib plots.
7. Compare Base vs RAG only.

### Acceptance Criteria

- Test dataset is immutable/versioned for the run.
- Raw model outputs are saved.
- Metrics can be recomputed from saved predictions.
- One command reproduces evaluation.
- Validation and test data are clearly separate.

---

## PHASE 7 — Fine-Tuning Dataset Preparation

### Goal

Create training data without contaminating the test benchmark.

### Tasks

1. Define instruction format.
2. Build training examples from approved domain sources.
3. Keep provenance for each example.
4. Deduplicate near-identical examples where possible.
5. Split train/validation before final test evaluation.
6. Validate JSONL/schema.
7. Measure:
   - example count
   - answer length distribution
   - question categories
8. Manually inspect a sample.
9. Do not include evaluation answers verbatim if they originate from the held-out set.

### Recommended Record

```json
{
  "id": "train_0001",
  "instruction": "Answer the domain question accurately.",
  "question": "...",
  "answer": "...",
  "source_document_id": "...",
  "metadata": {}
}
```

### Acceptance Criteria

- Dataset passes schema checks.
- No overlap with test IDs.
- Provenance is available.
- Human has reviewed a meaningful sample.
- Dataset summary is recorded.

---

## PHASE 8 — QLoRA Fine-Tuning

### Goal

Train a small adapter on the domain dataset while keeping the base model fixed.

### Tasks

1. Confirm the exact same base model as Base/RAG.
2. Load tokenizer and model.
3. Configure quantization only if supported by hardware/runtime.
4. Configure LoRA target modules based on the model architecture.
5. Define training config:
   - seed
   - epochs
   - batch size
   - gradient accumulation
   - learning rate
   - max sequence length
   - LoRA rank/alpha/dropout
6. Train on train split.
7. Evaluate on validation split.
8. Save only necessary adapter artifacts.
9. Save training metrics/config.
10. Implement adapter loading for inference.
11. Implement `FineTunedPipeline`.
12. Return the common `AnswerResult`.

### Rules

- Never alter the test set during tuning.
- Do not report validation metrics as test metrics.
- Do not hide failed runs.
- Do not change several hyperparameters at once without recording them.
- Avoid expensive broad sweeps; run a small number of justified experiments.

### Acceptance Criteria

- Adapter loads on top of the recorded base model.
- A clean process can reproduce inference.
- Training config is saved.
- Validation metrics are saved.
- Fine-tuned pipeline uses no FAISS in the baseline fine-tuned experiment.
- Unit/integration smoke tests pass.

---

## PHASE 9 — Final Comparative Evaluation

### Goal

Run the fair three-way experiment.

### Compared Systems

```text
A. Base LLM
B. Same Base LLM + RAG
C. Same Base LLM + LoRA/QLoRA adapter
```

### Frozen Inputs

Before the final test run, freeze:

- test dataset
- base model
- embedding model
- index build
- chunking config
- top-k
- prompt versions
- fine-tuned adapter
- generation config
- metric implementation

### Run Procedure

For every test item:

1. run Base
2. run RAG
3. run Fine-tuned
4. save raw outputs
5. save latency
6. save retrieved chunks for RAG
7. compute metrics after outputs are persisted

### Required Report Sections

1. Dataset description
2. Model/config description
3. Retrieval metrics
4. Answer quality metrics
5. Groundedness/hallucination analysis
6. Latency/resource comparison
7. Qualitative examples
8. Failure analysis
9. Limitations
10. Conclusion based only on measured results

### Important Research Rule

Do not force a narrative such as “RAG must win.”

Report what the measurements actually show.

### Acceptance Criteria

- All three systems ran on the same test set.
- Config is frozen and saved.
- Raw predictions are preserved.
- Summary metrics are reproducible.
- Failure cases are documented.
- No test-time manual editing of answers.

---

## PHASE 10 — FastAPI Backend + Persistence

### Goal

Expose stable application APIs after ML behavior is proven.

### Tasks

1. Create FastAPI app.
2. Implement dependency-initialized pipelines.
3. Add `/health`.
4. Add `/ask`.
5. Validate request with Pydantic schemas.
6. Persist:
   - query
   - response
   - pipeline
   - latency
   - model ID
   - prompt version
   - retrieval events for RAG
7. Return structured errors.
8. Add API tests with mocked lightweight model components.
9. Avoid reloading the model on every request.

### Acceptance Criteria

- API launches successfully.
- `/health` works.
- `/ask` works for all three pipeline names.
- Invalid pipeline returns a validation error.
- Results are written to SQLite.
- RAG source records can be traced back to chunks.
- Model is initialized predictably, not per token/request.

---

## PHASE 11 — Streamlit Frontend

### Goal

Create a clear demo, not a frontend engineering project.

### Required UI

1. Project/domain title
2. Question input
3. Pipeline selector:
   - Base
   - RAG
   - Fine-tuned
4. Ask button
5. Answer panel
6. Latency
7. RAG sources when applicable
8. Comparison/evaluation page
9. Clear error states

### Optional Nice-to-Have

“Compare all three” mode can call the backend three times and display responses side-by-side.

### Rules

- Streamlit never imports model training code.
- Streamlit never directly edits FAISS.
- Streamlit never loads the LLM.
- API URL is configurable.
- Do not spend excessive time on styling.

### Acceptance Criteria

A fresh user can:

1. open the app
2. ask one question
3. select each pipeline
4. see answer
5. see sources for RAG
6. view benchmark summary

---

## PHASE 12 — Hardening, Documentation, and Demo Release

### Goal

Make the project portfolio-ready.

### Tasks

1. Clean README.
2. Add architecture diagram.
3. Add setup instructions.
4. Add reproducible commands.
5. Add sample screenshots.
6. Add benchmark table with actual numbers.
7. Add limitations.
8. Add known failure modes.
9. Add dataset/source/license notes.
10. Add model/license notes.
11. Add security/privacy notes.
12. Remove secrets and local absolute paths.
13. Run full test suite.
14. Run from clean environment if possible.
15. Tag release.

### README Must Explain

- problem
- real-world use case
- architecture
- Base vs RAG vs Fine-tuning methodology
- dataset
- evaluation
- results
- how to run
- limitations
- future work

### Acceptance Criteria

- Clean clone can be set up from documentation.
- Core commands are documented.
- Demo works.
- Final benchmark is reproducible.
- Repository contains no keys or private data.
- CV claims match measured behavior.

---

# 11. Recommended Build Order Summary

```text
0  Guardrails + repo
1  Domain/data contract
2  Ingestion
3  Chunks + embeddings + FAISS
4  Base LLM
5  RAG
6  Evaluation harness (Base vs RAG)
7  Fine-tune dataset
8  QLoRA training
9  Final 3-way evaluation
10 FastAPI + SQLite persistence
11 Streamlit
12 Hardening + portfolio release
```

This ordering is intentional.

Do **not** build the UI first.
Do **not** fine-tune before an evaluation harness exists.
Do **not** add FastAPI complexity before the core pipelines work.

---

# 12. Testing Strategy

## 12.1 Unit Tests

Use small deterministic components for:

- cleaning
- chunking
- config parsing
- DB repositories
- vector-ID mapping
- prompt formatting
- metric calculation
- API validation

## 12.2 Integration Tests

Cover:

- document -> chunks -> SQLite
- chunks -> embeddings -> FAISS -> retrieval
- RAG pipeline with mocked/local lightweight LLM
- adapter load smoke test
- FastAPI -> pipeline -> SQLite

## 12.3 Real Model Smoke Tests

Keep these separate because they can be slow/GPU-dependent.

Do not require downloading a large model for every `pytest` run.

---

# 13. Error Handling Requirements

Failures must be explicit.

Examples:

- PDF has no extractable text
- unsupported file type
- missing FAISS index
- index built with different embedding model
- model unavailable
- adapter incompatible with base model
- CUDA OOM
- malformed dataset
- empty question
- missing SQLite row for retrieved vector ID

Do not use broad `except Exception: pass`.

Do not silently return fake answers when a component fails.

---

# 14. Logging

Log useful operational metadata, not entire private datasets.

Recommended:

- component initialized
- model IDs
- index version
- document IDs
- request ID
- pipeline
- timing
- exception context

Avoid logging:

- API keys
- full private documents
- secrets
- huge model outputs unnecessarily

---

# 15. Configuration Rules

Values that may change between experiments go in config, not hard-coded constants.

Example `config/rag.yaml`:

```yaml
embedding_model: TO_BE_DECIDED
chunk_size: TO_BE_DECIDED
chunk_overlap: TO_BE_DECIDED
top_k: TO_BE_DECIDED
index_path: artifacts/faiss/index.faiss
mapping_path: artifacts/faiss/mapping.json
prompt_version: rag_v1
```

Example `config/model.yaml`:

```yaml
base_model: TO_BE_DECIDED
max_new_tokens: 256
temperature: 0.0
adapter_path: artifacts/adapters/domain_adapter
```

The exact initial values are selected during the relevant phases.

---

# 16. Experiment Record Format

Every non-trivial experiment adds an entry to `docs/EXPERIMENT_LOG.md`.

Example:

```markdown
## EXP-003 — Chunking comparison

Date:
Git commit:
Dataset hash:
Embedding model:

Configurations:
- A: ...
- B: ...

Validation metrics:
...

Decision:
...

Reason:
...

Artifacts:
...
```

Do not overwrite old experiment outcomes.

---

# 17. Definition of Done for Any Coding Task

A task is not done because “the code was written.”

It is done when:

1. code is implemented
2. tests are added/updated
3. tests pass
4. relevant command was actually run
5. errors are handled
6. config/docs are updated if behavior changed
7. `PROJECT_STATE.md` is updated
8. no out-of-scope architecture was introduced
9. the next allowed task is stated

---

# 18. Agent Change Protocol

Before modifying code, the coding agent must state internally or in its work log:

```text
Current phase:
Current task:
Files inspected:
Files to modify:
Acceptance criteria:
Tests to run:
```

After modifying code:

```text
Files changed:
Commands run:
Test results:
Acceptance criteria status:
Known limitations:
PROJECT_STATE updated:
Next allowed task:
```

If the agent cannot verify something, it must say **UNVERIFIED**, not assume success.

---

# 19. Anti-Hallucination Rules for the Coding Agent

The coding agent must never:

1. claim a file exists without inspecting the repository
2. invent a package API from memory when uncertain
3. claim tests passed without running them
4. claim a model trains successfully without an actual run
5. invent benchmark numbers
6. invent dataset sizes
7. invent GPU memory consumption
8. change the locked stack silently
9. rewrite architecture because another framework is “easier”
10. add abstractions before they are needed
11. modify the held-out test set to improve results
12. present an LLM judge score as objective truth
13. hide failed experiments
14. create fake citations
15. silently fall back from RAG to base inference
16. silently use internet/API services not approved by the human

---

# 20. Security / Privacy Baseline

Even though this is a college project:

- secrets live in `.env`, never Git
- `.env.example` contains names, not real values
- validate uploaded file extension and size
- normalize paths; prevent path traversal
- do not execute uploaded files
- treat document text as untrusted data
- do not concatenate raw user text into SQL
- use parameterized SQLite queries
- do not claim production-grade security

---

# 21. Resource / Cost Strategy

Default to free/local resources.

- Local CPU for ingestion, FAISS, SQLite, tests.
- Local GPU if available for inference.
- Kaggle/Colab for QLoRA training if needed.
- Avoid paid APIs as baseline dependencies.
- Save adapters/checkpoints to prevent retraining.
- Keep early corpus and validation experiments small.

The project should remain usable even if no paid service is available.

---

# 22. Performance Strategy

Optimize only after correctness.

Measure first:

- ingestion time
- indexing time
- retrieval latency
- generation latency
- memory use
- index size

Do not introduce caching layers or async complexity before a measured bottleneck exists.

---

# 23. What Makes the Project Resume-Worthy

The final repository should demonstrate:

- end-to-end NLP system design
- document ingestion
- embeddings
- semantic retrieval
- vector indexing
- RAG
- open-source LLM inference
- PEFT/QLoRA fine-tuning
- experiment design
- no-test-leakage methodology
- quantitative evaluation
- backend API design
- SQL persistence
- frontend demo
- reproducibility and testing

The strength of the project is **not the number of frameworks**. It is the quality of the experiment and implementation.

---

# 24. Final Deliverables Checklist

## Product

- [ ] Base pipeline
- [ ] RAG pipeline
- [ ] Fine-tuned pipeline
- [ ] FastAPI backend
- [ ] SQLite persistence
- [ ] Streamlit UI
- [ ] RAG source citations

## Research

- [ ] documented corpus
- [ ] train/validation/test separation
- [ ] retrieval evaluation
- [ ] answer evaluation
- [ ] groundedness/hallucination analysis
- [ ] latency/resource comparison
- [ ] failure analysis
- [ ] reproducible final benchmark

## Engineering

- [ ] tests
- [ ] configuration files
- [ ] project state
- [ ] experiment log
- [ ] README
- [ ] no secrets
- [ ] clean Git history/release

---

# 25. First Action After Adding This Document

The coding agent should **not start implementing RAG immediately**.

The first action is:

1. inspect the existing repository
2. determine whether Phase 0 is already satisfied
3. create/update `docs/PROJECT_STATE.md`
4. create/update `docs/DECISIONS.md`
5. run existing tests
6. report the first unmet Phase 0 acceptance criterion
7. implement only enough to satisfy that criterion

Proceed sequentially from there.

---

# 26. Final Principle

> Build the smallest correct version, verify it, record what happened, then move to the next phase.

When there is a conflict between “moving fast” and preserving a fair, reproducible comparison, preserve the fair comparison.
