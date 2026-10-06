# Decisions

> **Rule:** Record every explicit architecture/configuration decision here.
> The coding agent must NOT silently choose any item marked `DECISION_REQUIRED`.

---

## Decision Template

```
### D-NNN — <title>
Date: YYYY-MM-DD
Phase: N
Status: APPROVED | PENDING | SUPERSEDED
Decision: <what was chosen>
Rationale: <why>
Alternatives considered: <what else was evaluated>
Constraints: <any limits or conditions>
```

---

## Technology Stack

### D-001 — Core technology stack locked

- **Date:** 2026-09-28
- **Phase:** 0
- **Status:** APPROVED
- **Decision:** The locked stack defined in `docs/IMPLEMENTATION_PLAN.md §1` is the authoritative technology list.
- **Rationale:** Established by the human project owner before implementation began.
- **Key items:** Python, PyTorch, Hugging Face Transformers, Sentence Transformers, FAISS, PEFT, TRL, LoRA/QLoRA, Hugging Face Datasets, Pandas, PyMuPDF, FastAPI, SQLite, Streamlit, scikit-learn, Matplotlib, pytest.
- **Explicitly excluded without approval:** LangChain, LlamaIndex, PostgreSQL, MongoDB, Redis, Celery, Kafka, Kubernetes, paid LLM APIs, agent frameworks, microservices.

### D-002 — Package name is `contextforge`

- **Date:** 2026-09-28
- **Phase:** 0
- **Status:** APPROVED
- **Decision:** The Python package/module is `contextforge`, located at `src/contextforge/`. The old planning-document name `DomainAdapt`/`domainadapt` is replaced by `ContextForge`/`contextforge` for all new source code.
- **Rationale:** Project was renamed from DomainAdapt to ContextForge before implementation started.

### D-003 — Project layout is `src/` layout

- **Date:** 2026-09-28
- **Phase:** 0
- **Status:** APPROVED
- **Decision:** Source lives under `src/contextforge/`. Build backend is setuptools with `[tool.setuptools.package-dir]` pointing to `src/`.
- **Rationale:** `src/` layout prevents accidental import of un-installed package during testing.

---

## Pending Decisions (Agent must stop and ask the human)

### D-P01 — Domain

- **Phase:** 1
- **Status:** APPROVED (2026-09-28)
- **Decision:** Domain is **cybersecurity knowledge and defensive security guidance**.
- **Rationale:** Bounded, high-value domain with multiple freely accessible authoritative documents (OWASP, NIST, CISA). Clear inclusion/exclusion boundary between defensive and offensive content.
- **Constraints:** Offensive security operations are explicitly out of scope.

### D-P02 — Corpus sources

- **Phase:** 1
- **Status:** APPROVED (2026-09-28)
- **Decision:** Initial corpus consists of three source families: OWASP Top 10:2025, NIST Cybersecurity Framework 2.0, and CISA Cross-Sector Cybersecurity Performance Goals v2.0.
- **Rationale:** All three are freely accessible, authoritative, complementary in coverage (web-app vulnerabilities, risk framework, performance goals), and suitable for defensive QA.
- **Constraints:** Additional corpus families require explicit human approval.

### D-P03 — Embedding model

- **Phase:** 3 (PHASE_3_DECISION)
- **Status:** APPROVED (2026-09-28)
- **Decision:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384 dimensions, L2-normalized).
- **Rationale:** Approved by human project owner. Lightweight, verified on local CPU (Python 3.13), strong dense retrieval performance for English technical domains.

### D-P04 — Base LLM

- **Phase:** 4 (PHASE_4_DECISION)
- **Status:** APPROVED (2026-09-28)
- **Decision:** `Qwen/Qwen2.5-1.5B-Instruct` as the single base model for ContextForge (Base, RAG, and subsequent PEFT/LoRA/QLoRA adaptation).
- **Rationale:** Approved by human project owner. Instruction-tuned causal LLM, ~1.54B parameters, manageable on local CPU/cloud hardware, native Hugging Face Transformers support, Apache-2.0 license.

### D-P05 — Chunking parameters

- **Phase:** 3 (PHASE_3_EXPERIMENT)
- **Status:** APPROVED (2026-09-28)
- **Decision:** Selected `medium` profile (`chunk_size_tokens=300`, `overlap_tokens=50`) via experiments EXP-001 and EXP-002.
- **Rationale:** Balanced context granularity (394 total chunks), 86.7% Recall@1, 100% Recall@3 and Recall@5, 0.933 MRR at 16.8ms latency.

### D-P06 — Top-k retrieval

- **Phase:** 5 (PHASE_5_EXPERIMENT)
- **Status:** APPROVED (2026-10-05)
- **Decision:** Selected `top_k = 5` as standard retrieval count for `RagPipeline` in conjunction with `medium` chunk profile and 1500-token context budget.
- **Rationale:** Evaluated K=3, 5, 8 in EXP-003 engineering comparison. K=3 (avg ~780 tokens) risks omitting secondary sections/context across multi-document sources. K=8 (avg ~1950 tokens) crowds or exceeds the 1500-token context budget, causing budget cutoffs and higher CPU latency with diminishing marginal relevance. K=5 (avg ~1220 tokens) fits comfortably within the budget, retrieves comprehensive cross-document evidence, and maintains high precision.
- **Constraints:** Development engineering baseline; can be tuned if evaluated on broader benchmark suites.

### D-P07 — Fine-tuning dataset format

- **Phase:** 7 (PHASE_7_DECISION)
- **Status:** APPROVED (2026-10-06)
- **Decision:** Chat-style JSONL with `system`, `user`, `assistant` messages. 55 hand-crafted grounded QA pairs, deterministic 70/15/15 split (38 train, 8 val, 9 test) via SHA-256 hash sort. No LLM-generated synthetic data.
- **Rationale:** Compatible with Qwen2.5 chat template and standard instruction-tuning pipelines (TRL SFTTrainer). Deterministic split ensures reproducibility. All answers grounded in approved corpus (OWASP, NIST, CISA).
- **Constraints:** Evaluation set (`evaluation_v1.jsonl`) questions are excluded from fine-tuning dataset to prevent test leakage.


### D-P08 — QLoRA hyperparameters

- **Phase:** 8 (PHASE_8_EXPERIMENT)
- **Status:** PENDING

### D-P09 — Final evaluation thresholds

- **Phase:** 9 (PHASE_9_DECISION)
- **Status:** PENDING

---

## Phase 1 Decisions

### D-004 — Project name is ContextForge

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** Official project name is **ContextForge**. All new source code, documentation, and configuration use `contextforge` (package) / `ContextForge` (display name). Planning documents originally written as DomainAdapt retain their internal naming where it does not affect architecture; technical architecture is unchanged.

### D-005 — Domain: cybersecurity knowledge and defensive security guidance

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** ContextForge answers questions grounded in authoritative defensive cybersecurity documentation.
- **Rationale:** Human decision. Bounded domain with clear in/out-of-scope distinction.
- **Constraints:** Offensive security operations (malware, exploit generation, credential attacks, phishing automation, autonomous pentesting) are explicitly out of scope.

### D-006 — Primary QA task: evidence-grounded domain-specific question answering

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** The primary task is single-turn, evidence-grounded QA. Questions must be answerable from the approved corpus. The system must acknowledge insufficient evidence rather than fabricate answers.
- **Rationale:** Aligns with the research goal of comparing base, RAG, and fine-tuned paths on a controlled, verifiable task.

### D-007 — Intended users

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** Primary intended users are: students learning cybersecurity, developers seeking secure-coding guidance, junior security professionals, and teams searching for authoritative defensive cybersecurity answers.

### D-008 — Initial corpus: OWASP Top 10:2025, NIST CSF 2.0, CISA CPGs v2.0

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** The three approved source families are recorded in `data/manifests/corpus_manifest.csv`.
- **Verified metadata:**
  - OWASP Top 10:2025: published November 2025; primary access at https://owasp.org/Top10; CC BY-SA 4.0
  - NIST CSF 2.0: published 2024-02-26; DOI https://doi.org/10.6028/NIST.CSWP.29; PDF https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf; public domain
  - CISA CPGs v2.0: published 2025-12-11; https://www.cisa.gov/cross-sector-cybersecurity-performance-goals; direct PDF https://www.cisa.gov/sites/default/files/2025-12/cpg-report-2.0.pdf; U.S. Government public domain
- **Pending at download time:** `retrieved_at`, `local_path`, `checksum` for each document.

### D-009 — QA question category taxonomy (7 categories)

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** Seven categories: `factual_definition`, `recommendation`, `control_guidance`, `framework_mapping`, `risk_explanation`, `mitigation`, `procedural_guidance`. Defined in `docs/DATASET_CONTRACT.md §2`.

### D-010 — Dataset split rules and leakage-prevention protocol

- **Date:** 2026-09-28
- **Phase:** 1
- **Status:** APPROVED
- **Decision:** Three-way split (train / validation / test). Test set is frozen before final evaluation and must never be used for tuning. Full rules in `docs/DATASET_CONTRACT.md §4–5`.
- **Key rule:** Document corpus overlap across splits is acceptable; QA-example identity leakage across the test boundary is not.

---

## Phase 2 Decisions

### D-011 — Ingestion layer supports PDF and Markdown formats

- **Date:** 2026-09-28
- **Phase:** 2
- **Status:** APPROVED
- **Decision:** The ingestion layer explicitly supports two document formats: PDF (`.pdf`) via PyMuPDF and Markdown (`.md`, `.markdown`) via a dedicated standard-library loader. Generic web crawlers, Word loaders, and HTML spiders are prohibited.
- **Rationale:** Aligns with the approved primary sources: NIST CSF 2.0 (PDF), CISA CPG 2.0 (PDF), and OWASP Top 10:2025 (official Markdown documentation).

### D-012 — Canonical OWASP Top 10:2025 uses official Markdown documentation

- **Date:** 2026-09-28
- **Phase:** 2
- **Status:** APPROVED
- **Decision:** The canonical OWASP Top 10:2025 corpus consists of the 16 official Markdown documentation files from the official repository (`OWASP/Top10/2025/docs/en/*.md`). The OWASP slide presentation PDF (`2025/Presentations/OWASP Top Ten 2025.pdf`) is not used as the canonical corpus.
- **Rationale:** The Markdown files contain the complete, authoritative textual vulnerability definitions, prevention controls, example attack scenarios, and mapped CWEs required for high-fidelity defensive QA.

### D-013 — Duplicate document detection enforced via artifact SHA-256

- **Date:** 2026-09-28
- **Phase:** 2
- **Status:** APPROVED
- **Decision:** Every ingested artifact is fingerprinted with its raw SHA-256 checksum. A UNIQUE constraint on `documents.checksum` in SQLite prevents duplicate records. The service and CLI default to skipping already-ingested artifacts with status `ALREADY_INGESTED`.
- **Rationale:** Ensures deterministic, idempotent ingestion and prevents duplicate chunks or inflated corpus statistics.

### D-014 — Conservative text cleaning preserving technical tokens

- **Date:** 2026-09-28
- **Phase:** 2
- **Status:** APPROVED
- **Decision:** Text cleaning performs Unicode NFKC normalization, whitespace collapsing, and control-character removal, but strictly preserves case, punctuation, markdown structure, code blocks, and technical cybersecurity identifiers (e.g. CVE identifiers, protocols, SQL syntax, XSS vectors, URLs).
- **Rationale:** Aggressive cleaning destroys the semantic meaning of code, vulnerability identifiers, and configuration commands essential for cybersecurity QA.

---

## Phase 3 Decisions

### D-015 — Dense vector search using FAISS Flat Inner Product
- **Date:** 2026-09-28
- **Phase:** 3
- **Status:** APPROVED
- **Decision:** Use exact `faiss.IndexFlatIP` (384-dimensional inner product). With L2-normalized embeddings, inner product is mathematically identical to cosine similarity. No approximate Voronoi cells (IVF) or lossy quantization (PQ) are used.
- **Rationale:** The entire authoritative corpus generates 394 chunks (medium) to 638 chunks (small). Exact search executes in <1ms within FAISS on CPU with zero recall degradation.

### D-016 — Configurable BGE query instruction prefix with domain tuning
- **Date:** 2026-09-28
- **Phase:** 3
- **Status:** APPROVED
- **Decision:** Support the official BGE query instruction prefix (`"Represent this sentence for searching relevant passages: "`) via `config/rag.yaml` (`query_instruction_enabled`), with passages strictly never receiving instructions.
- **Rationale:** Tested in EXP-002. Preserves the option for generic queries while allowing domain-specific technical queries to disable the prefix for higher Recall@1 (86.7% vs 73.3%).

---

## Phase 4 Decisions

### D-017 — Base LLM selection: Qwen/Qwen2.5-1.5B-Instruct
- **Date:** 2026-09-28
- **Phase:** 4
- **Status:** APPROVED
- **Decision:** Use `Qwen/Qwen2.5-1.5B-Instruct` as the single base causal language model across ContextForge (Base baseline, Phase 5 RAG generation, and Phase 7/8 LoRA fine-tuning).
- **Rationale:** Instruction-tuned open model, 1.54B parameters, native Transformers support, Apache-2.0 license, manageable locally in bfloat16 on CPU/GPU without aggressive quantization.

### D-018 — Deterministic baseline generation controls
- **Date:** 2026-09-28
- **Phase:** 4
- **Status:** APPROVED
- **Decision:** The baseline inference path enforces `do_sample: false` (greedy decoding), omitting temperature and top_p arguments.
- **Rationale:** Ensures that downstream comparisons between Base, RAG, and fine-tuned pipelines reflect genuine architectural differences rather than stochastic sampling noise.

### D-019 — Centralized baseline prompt versioning without corpus claims
- **Date:** 2026-09-28
- **Phase:** 4
- **Status:** APPROVED
- **Decision:** The base prompt (`base_v1`) defines a concise defensive cybersecurity persona and explicitly avoids claiming access to external documents or naming corpus frameworks unless requested by the user.
- **Rationale:** Prevents prompt-induced hallucination of document access in the baseline path where no retrieval context is provided.

