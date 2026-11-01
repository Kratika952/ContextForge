# ContextForge Architecture

> System architecture reference for the ContextForge domain-specific AI knowledge assistant.

---

## 1. High-Level Architecture

```
User (Browser)
     ↓
Streamlit Frontend (src/contextforge/ui/app.py)
     ↓  HTTP (JSON)
FastAPI Backend (src/contextforge/api/main.py)
     ↓
Pipeline Manager (src/contextforge/api/pipeline_manager.py)
     ↓  lazy load / unload
┌──────────────┬────────────────────┬───────────────────────┐
│  Base LLM    │   RAG Pipeline     │  Fine-Tuned Pipeline  │
│  (pipelines/ │  (pipelines/       │  (pipelines/          │
│   base.py)   │   rag.py)          │   finetuned.py)       │
└──────────────┴────────────────────┴───────────────────────┘
     ↓                                       ↓
AnswerResult (common/types.py)        SQLite Persistence
                                    (storage/db.py,
                                     storage/repositories.py)
```

**Key constraint:** The Streamlit frontend acts strictly as an HTTP client. It never imports pipeline code, model loaders, FAISS, or CUDA libraries.

---

## 2. Base Pipeline

**Path:** `src/contextforge/pipelines/base.py`

- Loads `Qwen/Qwen2.5-1.5B-Instruct` via `BaseModelLoader`
- Applies `base_v1` system prompt (defensive cybersecurity persona, no document claims)
- Deterministic greedy decoding (`do_sample=False`)
- Returns `AnswerResult` with answer text, latency, model ID, token usage
- No retrieval context, no sources

---

## 3. RAG Pipeline

**Path:** `src/contextforge/pipelines/rag.py`

1. **Embedding:** Question encoded via `BAAI/bge-small-en-v1.5` (384-dim, L2-normalized)
2. **Retrieval:** `DenseRetriever` queries FAISS `IndexFlatIP` for top-K chunks (default K=5)
3. **Context Formatting:** Retrieved chunks formatted with `[SOURCE_1]`, `[SOURCE_2]` labels, 1500-token budget
4. **Generation:** `Qwen/Qwen2.5-1.5B-Instruct` with `rag_v1` prompt (groundedness constraints, prompt-injection defense)
5. **Citation Validation:** Post-generation check of cited source labels against retrieved chunk set
6. **Result:** `AnswerResult` with answer, sources (doc_id, section, score, text), latency

**Data flow:**
```
Question → BGE Embed → FAISS Search → Top-K Chunks
    → Context Formatter → [SOURCE_N] labeled text
    → Qwen2.5 Generation → Citation Validator → AnswerResult
```

---

## 4. Fine-Tuned Pipeline

**Path:** `src/contextforge/pipelines/finetuned.py`

- Loads base `Qwen/Qwen2.5-1.5B-Instruct` + PEFT LoRA adapter from `artifacts/finetuning/phase8_pilot/`
- QLoRA pilot: rank 16, trained on 38 domain-grounded QA pairs
- Same generation strategy as Base (greedy, deterministic)
- Returns `AnswerResult` with `model_id: "Qwen/Qwen2.5-1.5B-Instruct+lora"`
- No retrieval context, no sources

---

## 5. FastAPI Layer

**Path:** `src/contextforge/api/main.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health, active pipeline, DB connectivity |
| `/ask` | POST | Submit question with pipeline selection, returns `AskResponse` |

- Request/response validation via Pydantic schemas (`src/contextforge/api/schemas.py`)
- Dependency injection for `PipelineManager` and `QueryResultRepository`
- Error handling: `InvalidQuestionError` → 400, `ModelLoadError` → 503, generic → 500

---

## 6. Pipeline Lifecycle

**Path:** `src/contextforge/api/pipeline_manager.py`

- **Lazy loading:** Pipelines instantiated on first request, not at server startup
- **Single-resident:** Only one pipeline loaded in GPU VRAM at a time
- **Safe switching:** When switching pipelines, the current pipeline is explicitly unloaded (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) before loading the new one
- **Design rationale:** NVIDIA RTX 2050 has 4 GB VRAM — cannot hold multiple 1.5B parameter models simultaneously

---

## 7. SQLite Persistence

**Paths:** `src/contextforge/storage/db.py`, `src/contextforge/storage/repositories.py`

- Schema: `queries` table (question, pipeline, answer, latency, model_id, timestamps), `retrieval_events` table (RAG source metadata per query)
- `QueryResultRepository` provides save/lookup/delete operations
- Database initialized at server startup via `init_db()`
- Runtime DB path: `data/contextforge.db` (gitignored)

---

## 8. Streamlit Frontend

**Paths:** `src/contextforge/ui/app.py`, `src/contextforge/ui/client.py`

- `ContextForgeClient`: stdlib `urllib.request` HTTP client (no ML imports)
- API URL configurable via `CONTEXTFORGE_API_URL` env var or sidebar input
- Two pages:
  - **Domain Assistant:** Pipeline selector, question input, answer display, latency metrics, RAG source expanders
  - **Research Results:** Frozen Phase 9 evaluation metrics, comparative table, key findings, retrieval quality metrics

---

## 9. Evaluation Pipeline

**Path:** `src/contextforge/evaluation/`

- **Token metrics:** Precision, Recall, F1 (whitespace-tokenized overlap)
- **Semantic similarity:** BGE cosine similarity between predicted and reference answers
- **Retrieval metrics:** Document Hit@K, MRR against gold document IDs
- **Citation metrics:** Citation presence rate, citation validity against retrieved chunks
- **Execution:** `scripts/evaluate_three_way.py` runs sequential Base → RAG → Fine-Tuned evaluation with per-pipeline CUDA cleanup
- **Phase 9 results:** Frozen in `artifacts/evaluation/phase9_metrics.json` and `phase9_raw_predictions.json`

---

## 10. Artifact Locations

| Artifact | Path | Gitignored |
|---|---|---|
| FAISS index | `artifacts/faiss/` | Yes |
| LoRA adapter | `artifacts/finetuning/phase8_pilot/` | Yes |
| Phase 9 metrics | `artifacts/evaluation/phase9_metrics.json` | Yes |
| Phase 9 predictions | `artifacts/evaluation/phase9_raw_predictions.json` | Yes |
| Corpus (raw) | `data/raw/` | Yes |
| Processed chunks | `data/processed/` | Yes |
| QA splits | `data/qa/` | No (tracked) |
| Runtime DB | `data/contextforge.db` | Yes |
| Config files | `config/` | No (tracked) |
