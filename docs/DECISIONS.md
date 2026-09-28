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
- **Status:** PENDING
- **What is needed:** Select a Sentence Transformers–compatible embedding model after a compatibility test on the target hardware.

### D-P04 — Base LLM

- **Phase:** 4 (PHASE_4_DECISION)
- **Status:** PENDING
- **What is needed:** Select a small open instruction model compatible with Transformers, LoRA/QLoRA, and available hardware.

### D-P05 — Chunking parameters

- **Phase:** 3 (PHASE_3_EXPERIMENT)
- **Status:** PENDING
- **What is needed:** Chunk size and overlap chosen via controlled experiment (A/B/C variants as specified in the plan).

### D-P06 — Top-k retrieval

- **Phase:** 5 (PHASE_5_EXPERIMENT)
- **Status:** PENDING

### D-P07 — Fine-tuning dataset format

- **Phase:** 7 (PHASE_7_DECISION)
- **Status:** PENDING

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

