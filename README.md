# ContextForge

> **Domain-Specific AI Knowledge Assistant** — comparing Base LLM, RAG, and QLoRA Fine-Tuned LLM for question answering over trusted enterprise knowledge.

**Current Domain:** Cybersecurity  
**Trusted Corpus:** OWASP Top 10:2025 · NIST CSF 2.0 · CISA CPG 2.0

---

## Why ContextForge

Organizations maintain large volumes of policies, frameworks, manuals, and technical standards. Employees waste significant time searching these documents manually. ContextForge enables **natural-language question answering** over trusted knowledge sources and provides a rigorous comparison of three increasingly sophisticated approaches.

---

## Research Question

For domain-specific question answering, how do:

1. **Base LLM** (zero-shot pretrained model)
2. **RAG** (retrieval-augmented generation with dense FAISS retrieval)
3. **QLoRA Fine-Tuned LLM** (parameter-efficient adapter training on domain QA pairs)

compare in answer quality, document grounding, retrieval effectiveness, latency, and resource requirements?

---

## Architecture

```
User
 ↓
Streamlit UI (src/contextforge/ui/app.py)
 ↓  HTTP
FastAPI Backend (src/contextforge/api/main.py)
 ↓
Pipeline Manager (lazy load, single-resident GPU model)
 ↓
┌──────────────┬────────────────────┬───────────────────────┐
│  Base LLM    │   RAG Pipeline     │  Fine-Tuned Pipeline  │
│  (Qwen2.5)   │  FAISS → Context   │  Qwen2.5 + LoRA      │
│              │  → Qwen2.5         │  adapter              │
└──────┬───────┴────────┬───────────┴───────────┬───────────┘
       └────────────────┴───────────────────────┘
                        ↓
                  AnswerResult
                        ↓
                  SQLite Persistence
```

**RAG retrieval path:**
Question → BGE Embedding → FAISS IndexFlatIP → Top-K Chunks → Formatted Context → Qwen2.5 Generation → Citation Validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.10 |
| ML / DL | PyTorch |
| LLM Runtime | Hugging Face Transformers (`Qwen/Qwen2.5-1.5B-Instruct`) |
| Embeddings | Sentence Transformers (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Vector Search | FAISS (`IndexFlatIP`, exact cosine via L2-normalized embeddings) |
| Fine-Tuning | PEFT + TRL + LoRA/QLoRA |
| Dataset Utilities | Hugging Face Datasets + Pandas |
| PDF Extraction | PyMuPDF |
| Backend API | FastAPI + Uvicorn |
| Persistence | SQLite |
| Frontend | Streamlit |
| Testing | pytest |
| Configuration | YAML + environment variables |

---

## Models

| Role | Model | Notes |
|---|---|---|
| Base / Generation LLM | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B params, Apache-2.0, BF16 |
| Embedding Model | `BAAI/bge-small-en-v1.5` | 384-dim, L2-normalized |
| Fine-Tuned Adapter | QLoRA pilot (LoRA rank 16) | 38 train examples, 8 val, 9 test |

---

## Final Research Results

> **Benchmark:** 9 held-out supported cybersecurity questions, BF16 precision, CUDA, sequential pipeline execution.

| Metric | Base LLM | RAG | Fine-Tuned |
|---|---|---|---|
| **Token F1** | 0.2870 | 0.3159 | **0.3274** |
| **Semantic Similarity** | 0.8641 | **0.8715** | 0.8692 |
| **Mean Latency (ms)** | 8784.4 | 11837.3 | **5633.3** |

**RAG Retrieval Quality:**

| Metric | Value |
|---|---|
| Hit@5 | 1.0000 |
| MRR | 0.7722 |
| Citation Presence | 0.3333 |
| Citation Validity (emitted) | 1.0000 |

**Interpretation:**

- **Fine-Tuned** achieved the highest token F1 (0.3274) and lowest mean latency (5633.3 ms), producing concise domain-adapted answers.
- **RAG** achieved the highest semantic similarity (0.8715) and provides explicit document grounding with 100% citation validity on emitted citations.
- **Base LLM** serves as the unadapted comparison baseline.
- No single method dominated across all questions.

---

## Limitations

- Final held-out benchmark contains only **9 supported cybersecurity questions** — insufficient for production-level statistical validation.
- No unsupported/out-of-domain items in the Phase 9 benchmark (refusal metrics = NOT_APPLICABLE).
- Pilot fine-tuning dataset: only **38 training examples** (55 total QA pairs).
- Evaluated on a single GPU (NVIDIA RTX 2050, 4 GB VRAM) with CPU/GPU resource constraints.
- Results represent a **research prototype**, not production-level validation.

---

## Project Structure

```
ContextForge/
├── config/                # YAML configuration (model, RAG, finetune, evaluation)
├── data/                  # Corpus manifests, QA splits (raw docs gitignored)
├── docs/                  # Architecture, decisions, experiments, project state
├── notebooks/             # Phase 8 QLoRA pilot notebook
├── scripts/               # CLI entry points (ingest, build_index, train, evaluate, smoke tests)
├── src/contextforge/      # Python package (src layout)
│   ├── api/               # FastAPI backend (main, schemas, pipeline_manager)
│   ├── common/            # Shared types, config, error hierarchy
│   ├── evaluation/        # Token F1, semantic similarity, retrieval metrics
│   ├── ingestion/         # PDF/Markdown loaders, text cleaner, token-aware chunker
│   ├── llm/               # Model loader, prompts, generation wrapper
│   ├── pipelines/         # BasePipeline, RagPipeline, FineTunedPipeline
│   ├── retrieval/         # BGE embedder, FAISS vector store, dense retriever
│   ├── storage/           # SQLite schema, repositories
│   └── ui/                # Streamlit frontend, HTTP client
├── tests/
│   ├── unit/              # 145 unit tests
│   └── integration/       # End-to-end tests
├── artifacts/             # FAISS indices, adapters, evaluation results (gitignored)
├── pyproject.toml         # Project metadata and dependencies
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Kratika952/ContextForge.git
cd ContextForge

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. (First time) Ingest corpus documents
python scripts/ingest.py data/raw/nist-csf-2.0.pdf
python scripts/ingest.py data/raw/cisa-cpg-2.0.pdf
python scripts/ingest.py data/raw/owasp-top10-2025

# 5. Build FAISS retrieval index
python scripts/build_index.py --profile medium
```

> **GPU Note:** For CUDA inference and fine-tuning, install PyTorch with CUDA support separately before `pip install -e ".[dev]"`. A separate `.venv-gpu` environment may be used for GPU workloads.

---

## Run Backend

```bash
python -m uvicorn contextforge.api.main:app --host 127.0.0.1 --port 8000
```

---

## Run Frontend

```bash
streamlit run src/contextforge/ui/app.py --server.port 8501
```

The Streamlit UI provides:
- Pipeline selector (Base / RAG / Fine-Tuned)
- Question input with Ask action
- Answer display with latency and token usage
- RAG source documents with relevance scores
- Research Results dashboard with frozen Phase 9 metrics

---

## Run Tests

```bash
python -m pytest tests/unit/ -v --tb=short
```

Expected: **145 passed**, 0 failed.

---

## Evaluation

Phase 9 evaluation artifacts are stored in `artifacts/evaluation/`:
- `phase9_metrics.json` — aggregate metrics for all three pipelines
- `phase9_raw_predictions.json` — per-question predictions and scores

These are frozen results from a completed evaluation. **Do not rerun** — the evaluation script (`scripts/evaluate_three_way.py`) requires CUDA and the full model/index/adapter stack.

---

## Security Scope

ContextForge answers questions about **defensive cybersecurity** frameworks and best practices only.

**Out of scope** (and actively refused):
- Malware creation or analysis for offensive purposes
- Exploit generation or weaponization
- Credential theft techniques
- Phishing automation
- Any destructive or offensive automation

---

## Future Work

- Larger fine-tuning dataset (hundreds of grounded QA pairs)
- Larger held-out benchmark (50+ diverse questions including OOD/refusal items)
- Retrieval reranking (cross-encoder or ColBERT)
- Improved citation generation in RAG answers
- Multi-domain evaluation (compliance, healthcare, legal)
- Deployment optimization (quantization, model caching, batched inference)
- Production model lifecycle and versioning

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture reference |
| [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) | Step-by-step demo walkthrough |
| [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) | Portfolio-oriented project summary |
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
| 5 | RAG Pipeline | **COMPLETE** |
| 6 | Evaluation Harness v1 | **COMPLETE** |
| 7 | Fine-Tuning Dataset Preparation | **COMPLETE** |
| 8 | QLoRA Fine-Tuning | **COMPLETE** |
| 9 | Full Three-Way Evaluation | **COMPLETE** |
| 10 | FastAPI Backend | **COMPLETE** |
| 11 | Streamlit Frontend | **COMPLETE** |
| 12 | Hardening, Documentation, Demo Release | **COMPLETE** |

---

## Licence

MIT — see [LICENSE](LICENSE) if present.
