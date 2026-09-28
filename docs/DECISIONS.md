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

- **Phase:** 1 (HUMAN_DECISION_REQUIRED)
- **Status:** PENDING
- **What is needed:** The human must select a bounded domain for the QA system (e.g., software documentation, cybersecurity guidance, university regulations, research literature).
- **Blocking:** Phase 1 cannot start until this is resolved.

### D-P02 — Corpus sources

- **Phase:** 1 (HUMAN_DECISION_REQUIRED)
- **Status:** PENDING
- **What is needed:** Specific document sources, licensing confirmation, and at least 3 representative documents.
- **Blocking:** Phase 1 cannot start until this is resolved.

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
