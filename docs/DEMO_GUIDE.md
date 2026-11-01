# ContextForge Demo Guide

> Step-by-step walkthrough for demonstrating ContextForge's three inference pipelines and research evaluation dashboard.

---

## Prerequisites

- ContextForge installed (`pip install -e ".[dev]"`)
- Corpus ingested and FAISS index built (see README Setup)
- GPU environment available (`.venv-gpu`) for CUDA inference

---

## 1. Start the FastAPI Backend

```bash
# Use the GPU virtual environment for model inference
.venv-gpu\Scripts\python.exe -m uvicorn contextforge.api.main:app --host 127.0.0.1 --port 8000
```

Wait for: `Uvicorn running on http://127.0.0.1:8000`

Verify health:
```bash
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","service":"contextforge",...}`

---

## 2. Start the Streamlit Frontend

In a separate terminal:

```bash
.venv\Scripts\streamlit.exe run src/contextforge/ui/app.py --server.port 8501
```

Open: **http://localhost:8501**

---

## 3. Demo Flow

### 3a. Test API Connection

1. In the sidebar, click **"Test API Connection"**
2. Verify green success message: `Connected (status: ok)`

### 3b. Ask a Base LLM Question

1. Select **"Base LLM"** from the pipeline dropdown
2. Enter: **"What does least privilege mean in cybersecurity?"**
3. Click **"Ask Assistant"**
4. Observe:
   - Answer appears (general cybersecurity knowledge, no documents cited)
   - Latency displayed (~5–15 seconds on first load)
   - No "Retrieved Sources" section

### 3c. Ask a RAG Question

1. Select **"RAG Pipeline"** from the dropdown
2. Enter: **"What is the Govern function in NIST CSF 2.0?"**
3. Click **"Ask Assistant"**
4. Observe:
   - Answer appears grounded in retrieved documents
   - Latency displayed (higher due to retrieval + generation)
   - **Retrieved Sources** section shows 5 document chunks with:
     - Document ID
     - Section name
     - Relevance score
     - Text excerpt (expandable)

**Key talking point:** RAG provides explicit source attribution — the user can verify the answer against the original document.

### 3d. Ask a Fine-Tuned Question

1. Select **"Fine-Tuned LLM"** from the dropdown
2. Enter: **"How does multi-factor authentication help secure accounts?"**
3. Click **"Ask Assistant"**
4. Observe:
   - Answer appears (concise, domain-adapted style)
   - Latency displayed (typically fastest of the three)
   - Model ID shows `Qwen/Qwen2.5-1.5B-Instruct+lora`
   - No sources (fine-tuned model generates from learned knowledge)

**Key talking point:** Fine-tuning produces faster, more concise answers but without explicit document grounding.

### 3e. View Research Results

1. In the sidebar, click **"📊 Research Results"**
2. Walk through:
   - **Comparative Performance Summary** table (Token F1, Semantic Similarity, Latency)
   - **Key Scientific Findings** cards showing each pipeline's strengths
   - **RAG Retrieval and Citation Quality** metrics (Hit@5: 100%, MRR: 0.7722)
   - **Interpretation** section summarizing tradeoffs
3. Note the **benchmark limitation** warning: 9 held-out questions only

---

## 4. Key Talking Points

| Topic | Point |
|---|---|
| **Architecture** | Decoupled: Streamlit → FastAPI → Pipeline Manager. Frontend never loads models. |
| **GPU Memory** | Single-resident pipeline design — only one model in 4 GB VRAM at a time. |
| **Fine-Tuned Strengths** | Highest token F1 (0.3274), lowest latency (5633 ms), concise outputs. |
| **RAG Strengths** | Highest semantic similarity (0.8715), explicit document grounding, 100% citation validity. |
| **Limitations** | 9-question benchmark, 38-example fine-tuning dataset, single GPU. |
| **No method dominates** | Each approach has distinct tradeoffs — the system lets users choose. |

---

## 5. Safe Demo Questions

Use these non-benchmark questions only:

- "What does least privilege mean in cybersecurity?"
- "What is the Govern function in NIST CSF 2.0?"
- "How does multi-factor authentication help secure accounts?"
- "What are common web application vulnerabilities?"
- "How does NIST CSF 2.0 organize its core functions?"

**Do NOT use the 9 Phase 9 held-out benchmark questions during demos.**

---

## 6. Troubleshooting

| Issue | Solution |
|---|---|
| Connection refused | Ensure FastAPI is running on port 8000 |
| Slow first response | First request loads model into GPU — subsequent requests are faster |
| CUDA OOM | Ensure only one pipeline is active; restart server if needed |
| Streamlit won't start | Check `streamlit` is installed in the active environment |
