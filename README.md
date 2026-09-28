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

**Phase 4 — Base LLM Pipeline** (complete, awaiting Phase 5 approval)

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the current execution state.

---

## Baseline LLM Inference

- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ~1.54B parameters)
- **Prompt Version:** `base_v1` (concise defensive cybersecurity persona, zero document access claims)
- **Generation Strategy:** Deterministic greedy decoding (`do_sample: false`, temperature/top_p omitted)
- **Runtime Strategy:** Auto device placement (CUDA or CPU), `torch.bfloat16` precision, eval mode, loaded once per service/process.
- **Inference CLI:**
  ```bash
  python scripts/run_base.py "What is broken access control?"
  ```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.10 |
| ML / DL | PyTorch |
| LLM runtime | Hugging Face Transformers (`Qwen/Qwen2.5-1.5B-Instruct`) |
| Embeddings | Sentence Transformers (`BAAI/bge-small-en-v1.5`) |
| Vector search | FAISS (`IndexFlatIP`) |
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

# 4. Ingest authoritative corpus
python scripts/ingest.py data/raw/nist-csf-2.0.pdf
python scripts/ingest.py data/raw/cisa-cpg-2.0.pdf
python scripts/ingest.py data/raw/owasp-top10-2025

# 5. Build FAISS dense retrieval indices
python scripts/build_index.py --profile medium

# 6. Run semantic search
python scripts/search.py "What is the NIST CSF Govern function?"

# 7. Run baseline LLM inference
python scripts/run_base.py "What is broken access control?"

# 8. Run test suite
pytest -v
```

---

## Repository Layout

```
ContextForge/
├── docs/                  # Implementation plan, decisions, state, experiment log
├── config/                # project.yaml, rag.yaml, model.yaml
├── src/contextforge/      # Python package (src layout)
│   ├── common/            # Shared types, config, error hierarchy
│   ├── ingestion/         # PDF, Markdown loaders, text cleaner, token-aware chunker
│   ├── storage/           # SQLite schema, db connection manager, repositories
│   ├── retrieval/         # BGE embedder, FAISS vector store, dense retriever
│   ├── llm/               # Model loader, prompts, generation wrapper
│   └── pipelines/         # BasePipeline (and future RagPipeline)
├── tests/                 # pytest test suite
│   ├── unit/              # Isolated unit tests with mocked models
│   └── integration/       # End-to-end retrieval and live LLM tests
├── data/                  # Corpora, manifests, and QA splits
├── artifacts/             # FAISS indexes, adapters, evaluation logs (gitignored)
└── scripts/               # CLI entry points (ingest, build_index, search, run_base)
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
| 2 | Document Ingestion | **COMPLETE** |
| 3 | Chunking, Embeddings, FAISS | **COMPLETE** |
| 4 | Base LLM Pipeline | **COMPLETE** |
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
