# ContextForge

> **Domain-specific QA system** comparing three inference paths:
> Base LLM · RAG · Fine-tuned LLM (LoRA/QLoRA)

## Research Question

For domain-specific question answering, how do a base pretrained LLM,
retrieval-augmented generation, and parameter-efficient fine-tuning compare
in answer quality, groundedness/hallucination behaviour, retrieval quality,
latency, and resource requirements?

---

## Project Status

**Phase 0 — Repository, Environment, and Guardrails** (active)

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the current execution state.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.10 |
| ML / DL | PyTorch |
| LLM runtime | Hugging Face Transformers |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Fine-tuning | PEFT + TRL + LoRA/QLoRA |
| Quantization | bitsandbytes (where supported) |
| Dataset utilities | Hugging Face Datasets + Pandas |
| PDF extraction | PyMuPDF |
| Backend API | FastAPI |
| App database | SQLite |
| Frontend | Streamlit |
| Metrics | scikit-learn + custom evaluation |
| Visualisations | Matplotlib |
| Testing | pytest |
| Configuration | env vars + YAML/JSON config |

---

## Quickstart

```bash
# 1. Clone the repository
git clone <repo-url>
cd ContextForge

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package in editable mode (dev extras)
pip install -e ".[dev]"

# 4. Copy and populate the environment file
cp .env.example .env
# Edit .env — see comments inside for required variables

# 5. Run the smoke tests
pytest
```

---

## Repository Layout

```
ContextForge/
├── docs/                  # Implementation plan, decisions, state
├── config/                # project.yaml and future phase configs
├── src/contextforge/      # Python package (src layout)
├── tests/                 # pytest test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/                  # Corpora and QA splits (mostly gitignored)
├── artifacts/             # FAISS indexes, adapters, plots (gitignored)
├── scripts/               # CLI entry points (populated in later phases)
└── frontend/              # Streamlit app (Phase 10+)
```

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Phase-by-phase technical plan |
| [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) | Current execution state |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Approved architecture decisions |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Experiment records |

---

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Repository, Environment, Guardrails | **COMPLETE** |
| 1 | Domain and Dataset Contract | **COMPLETE** |
| 2 | Document Ingestion | PLANNED |
| 3 | Chunking, Embeddings, FAISS | PLANNED |
| 4 | Base LLM Pipeline | PLANNED |
| 5 | RAG Pipeline | PLANNED |
| 6 | Evaluation Harness v1 | PLANNED |
| 7 | Fine-Tuning Dataset Preparation | PLANNED |
| 8 | QLoRA Fine-Tuning | PLANNED |
| 9 | Full Evaluation (3-way) | PLANNED |
| 10 | FastAPI Backend | PLANNED |
| 11 | Streamlit Frontend | PLANNED |
| 12 | Reporting and Packaging | PLANNED |

---

## Licence

MIT — see [LICENSE](LICENSE) if present.
