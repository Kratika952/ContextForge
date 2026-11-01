# ContextForge — Project Summary & Engineering Review

> **Domain-Specific AI Knowledge Assistant**  
> Comparing Base LLM, Retrieval-Augmented Generation (RAG), and QLoRA Fine-Tuning for Enterprise Domain Question Answering.

---

## 1. Problem Statement

Enterprises and regulated institutions maintain hundreds of pages of mission-critical standards, compliance guidelines, operational manuals, and architecture frameworks. When personnel need answers to specific questions, manual searching across PDFs and policy libraries is slow, error-prone, and inconsistently documented.

Generic large language models hallucinate domain details, lack access to private or newly updated documents, and cannot cite authoritative sources with confidence.

---

## 2. Solution: ContextForge

ContextForge is a modular, domain-specific AI knowledge assistant built to address this challenge. It provides:
1. **Natural-language question answering** over authoritative reference documents.
2. An **empirical, three-way comparison engine** contrasting:
   - **Base LLM** (zero-shot baseline)
   - **RAG** (dense semantic retrieval + grounded generation + citation verification)
   - **QLoRA Fine-Tuned LLM** (parameter-efficient domain adapter)
3. **Decoupled production architecture**: A lightweight FastAPI backend with single-resident GPU lifecycle management and a Streamlit interactive frontend.

**Current Implementation Domain:** Defensive Cybersecurity  
**Corpus:** OWASP Top 10:2025, NIST CSF 2.0, and CISA Cross-Sector Cybersecurity Performance Goals (CPG 2.0).

---

## 3. Research Design

To evaluate the three paradigms fairly without introducing confounding variables:

- **Strict Model Pairing:** All three approaches use the **exact same base model** (`Qwen/Qwen2.5-1.5B-Instruct`). The fine-tuned variant trains only low-rank adapter weights (LoRA rank $r=16$) on top of this identical base.
- **Controlled Retrieval:** RAG uses dense vector retrieval with `BAAI/bge-small-en-v1.5` embeddings (384-dimensional, L2-normalized) indexed in FAISS (`IndexFlatIP`).
- **No Test Leakage:** The evaluation set (9 held-out cybersecurity questions with reference ground truth answers) was strictly isolated from training, validation, chunk tuning, and prompt engineering.
- **Unified Evaluation Pipeline:** Automated scoring measures:
  - Lexical overlap (**Token F1**)
  - Semantic alignment (**Cosine similarity** via sentence embeddings)
  - End-to-end inference **latency** (milliseconds)
  - Retrieval effectiveness (**Hit@K**, **MRR**, citation precision & validity)

---

## 4. Dataset & Corpus

| Asset | Source / Description | Metrics |
|---|---|---|
| **Trusted Corpus** | OWASP Top 10:2025, NIST CSF 2.0, CISA CPG 2.0 | 3 authoritative PDFs, extracted via PyMuPDF |
| **Processed Chunks** | Structural + semantic chunking (size=500, overlap=50) | 185 chunks stored in SQLite + FAISS index |
| **Fine-Tuning Dataset** | Multi-document domain QA pairs | 38 train pairs, 8 validation pairs, 9 test pairs |

---

## 5. System Architecture

```
User Browser
     ↓
Streamlit UI (src/contextforge/ui/app.py)
     ↓  HTTP REST
FastAPI Backend (src/contextforge/api/main.py)
     ↓
Pipeline Manager (Single-resident GPU manager)
     ↓  loads on demand, unloads previous model, calls torch.cuda.empty_cache()
┌───────────────────┬──────────────────────────┬────────────────────────┐
│     Base LLM      │       RAG Pipeline       │   Fine-Tuned Pipeline  │
│  Qwen2.5-1.5B     │   FAISS IndexFlatIP      │   Qwen2.5-1.5B +       │
│  Instruct (BF16)  │   + BGE-small-en-v1.5    │   PEFT LoRA Adapter    │
│                   │   + Qwen2.5-1.5B         │                        │
└─────────┬─────────┴─────────────┬────────────┴───────────┬────────────┘
          └───────────────────────┼────────────────────────┘
                                  ↓
                             AnswerResult
                                  ↓
                        SQLite Persistence
                    (queries, runs, evaluations)
```

### Key Engineering Principles
- **Separation of Concerns:** Streamlit acts strictly as an HTTP client. It contains no torch, transformers, FAISS, or model code.
- **VRAM Constrained Operation:** On a 4GB NVIDIA RTX 2050 laptop GPU, running multiple LLMs simultaneously causes CUDA Out-Of-Memory (OOM). The `PipelineManager` ensures only one model is resident in VRAM at any given time, systematically clearing caches and triggering garbage collection between switches.
- **Separate Index and Metadata:** FAISS holds only dense vector embeddings; all document metadata, chunk text, and source references reside in SQLite.

---

## 6. Final Research Results (Phase 9 Frozen)

Tested on the held-out 9-question cybersecurity benchmark (BF16 precision, CUDA, greedy decoding):

| Metric | Base LLM | RAG | Fine-Tuned (QLoRA) |
|---|---|---|---|
| **Token F1** | 0.2870 | 0.3159 (+10.1%) | **0.3274** (+14.1%) |
| **Semantic Similarity** | 0.8641 | **0.8715** (+0.9%) | 0.8692 (+0.6%) |
| **Mean Latency (ms)** | 8,784.4 | 11,837.3 | **5,633.3** (-35.9%) |

### Retrieval Metrics (RAG)
- **Hit@5:** 1.0000 (100% of ground-truth sources retrieved in top 5 chunks)
- **MRR (Mean Reciprocal Rank):** 0.7722
- **Citation Presence:** 0.3333
- **Citation Validity (Emitted Citations):** 1.0000 (0% hallucinated sources)

### Interpretation & Tradeoffs
1. **Fine-Tuned LLM:** Delivers highest Token F1 and fastest response time (no retrieval latency overhead, concise domain style). However, weights are frozen after training and cannot cite live or updated document references.
2. **RAG:** Delivers highest semantic similarity and authoritative groundedness. Transparently exposes retrieved passages and citations for auditing, though incurring retrieval latency and context window overhead.
3. **Base LLM:** Provides a strong general baseline, but generates generic responses lacking specific compliance identifiers or structural citations.

---

## 7. Key Engineering Decisions & Challenges Solved

1. **4GB VRAM GPU Stability:**
   - *Problem:* Running Qwen2.5-1.5B alongside embeddings and adapter weights on a 4GB RTX 2050 caused intermittent CUDA OOM errors.
   - *Solution:* Implemented `PipelineManager` with explicit lifecycle management (`del self._active_pipeline`, `torch.cuda.empty_cache()`, `gc.collect()`), running at BF16 precision.

2. **Accurate Cosine Similarity with FAISS:**
   - *Problem:* Default Euclidean distance or raw dot-products do not guarantee normalized cosine metrics.
   - *Solution:* L2-normalized all chunk embeddings upon generation and used `faiss.IndexFlatIP` (inner product), mathematically guaranteeing exact cosine similarity.

3. **Prompt Injection & Source Boundary Defense:**
   - *Problem:* RAG contexts can inadvertently inject adversarial instructions from ingested PDFs.
   - *Solution:* Enforced strict `[SOURCE_N]` delimiters, an immutable system prompt instructing the model to treat context as raw reference data, and post-generation citation validation.

4. **Zero-Leakage Test Isolation:**
   - *Problem:* Accidental contamination of benchmark data during dataset generation or tuning.
   - *Solution:* Created independent SHA-256 verified splits (`train`, `val`, `test`), ensuring test items were excluded from prompt iterations and hyperparameter searches.

---

## 8. Limitations

- **Small Benchmark Sample:** The final held-out test suite contains 9 cybersecurity questions. While sufficient for pilot comparison, larger statistical validity requires 100+ multi-scenario questions.
- **Pilot Fine-Tuning Dataset:** 38 training pairs is a pilot dataset designed to prove adapter convergence and domain styling rather than exhaustive domain memorization.
- **Citation Frequency:** While emitted citations were 100% valid, the model emitted explicit `[SOURCE_N]` tags in 33% of answers, indicating further instruction tuning on citation formatting is beneficial.
- **Single Host Deployment:** The current system is architected for single-workstation local inference rather than distributed cluster serving.

---

## 9. Key Takeaways & What I Learned

- **RAG vs. Fine-Tuning is not an either/or choice:** They serve complementary roles. Fine-tuning aligns tone, terminology, and structural formatting; RAG supplies dynamic, verifiable facts and eliminates knowledge staleness.
- **Evaluation rigor is paramount:** High generation fluency often masks hallucination. Rigorous metrics (Token F1, semantic similarity, Hit@K, citation validity) provide objective ground truth.
- **Hardware constraints drive software quality:** Engineering for a 4GB GPU forced clean resource boundaries, decoupled microservices, and lazy-loading patterns that make the entire system more robust.

---

## 10. Future Work

- **Hybrid RAG + Fine-Tuned Pipeline:** Combine the QLoRA adapter with the RAG retrieval flow to evaluate whether combined domain style and live retrieval outperform either independently.
- **Cross-Encoder Reranking:** Integrate a lightweight cross-encoder (e.g., `bge-reranker-base`) after initial FAISS retrieval to improve Top-3 precision.
- **Multi-Domain Expansion:** Generalize corpus ingestion and evaluation to healthcare (HIPAA) and financial compliance (SOX) domains.
- **Quantization Serving:** Benchmark GGUF/AWQ 4-bit serving via vLLM or llama.cpp for higher concurrency.
