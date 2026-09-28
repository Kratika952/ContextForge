# Experiment Log

> **Rule:** Every experiment that compares quality or performance receives a record here.
> Do NOT overwrite old records. Append new entries only.
> Do NOT invent metrics. Only record values produced by actual runs.

---

## Log Format

```
### EXP-NNN — <title>
Date: YYYY-MM-DD
Phase: N
Purpose: <why this experiment was run>
Config:
  - key: value
Model IDs:
  - base_llm: <id or PENDING>
  - embedding_model: <id or PENDING>
Dataset:
  - split: <train|validation|test>
  - version/hash: <hash or description>
Seed: <integer or N/A>
Metrics:
  - <metric>: <value>
Conclusion: <what was learned>
Artifact paths:
  - <path or NONE>
Notes:
  - <any caveats>
```

---

## Phase 0 — No experiments yet

Phase 0 is a repository/environment setup phase. No ML experiments are conducted.
The first experiments will be recorded in Phase 3 (chunking strategy comparison)
and Phase 6 (Base vs RAG evaluation).

---

<!-- Append new experiment records below this line -->

### EXP-001 — Chunking Profile Comparison (Small vs Medium vs Large)
Date: 2026-09-28
Phase: 3
Purpose: Compare dense retrieval recall and latency across three controlled chunking profiles (small: 180/30, medium: 300/50, large: 420/70) on authoritative cybersecurity corpus (NIST CSF 2.0, CISA CPG 2.0, OWASP Top 10:2025).
Config:
  - embedding_model: "BAAI/bge-small-en-v1.5"
  - dimension: 384
  - metric: "inner_product (cosine similarity on L2-normalized vectors)"
  - index_type: "faiss.IndexFlatIP"
  - top_k: 5
  - query_instruction: true ("Represent this sentence for searching relevant passages: ")
Model IDs:
  - base_llm: PENDING
  - embedding_model: "BAAI/bge-small-en-v1.5"
Dataset:
  - split: smoke
  - version/hash: data/qa/retrieval_smoke.jsonl (15 grounded questions)
Seed: N/A (deterministic exact search)
Metrics:
  - small (180/30 tokens):
      total_chunks: 638
      recall@1: 0.733
      recall@3: 1.000
      recall@5: 1.000
      mrr: 0.844
      avg_latency_ms: 20.9
  - medium (300/50 tokens):
      total_chunks: 394
      recall@1: 0.733
      recall@3: 0.867
      recall@5: 1.000
      mrr: 0.833
      avg_latency_ms: 19.2
  - large (420/70 tokens):
      total_chunks: 281
      recall@1: 0.733
      recall@3: 0.800
      recall@5: 1.000
      mrr: 0.813
      avg_latency_ms: 22.7
Conclusion: All three profiles achieve 100% Recall@5. Small (180 tokens) yields highest Recall@3 (1.000) under standard query instructions, while medium (300 tokens) provides faster retrieval (19.2ms) with fewer, more context-rich passages (394 vs 638).
Artifact paths:
  - artifacts/faiss/small
  - artifacts/faiss/medium
  - artifacts/faiss/large
Notes:
  - Token counts verified via BGE AutoTokenizer. Zero empty chunks generated.
  - Chunks stored in SQLite and indexed in FAISS.

---

### EXP-002 — Query Instruction Prefix Ablation
Date: 2026-09-28
Phase: 3
Purpose: Measure impact of BGE query instruction prefix ("Represent this sentence for searching relevant passages: ") vs raw domain query text across all chunk profiles.
Config:
  - embedding_model: "BAAI/bge-small-en-v1.5"
  - dimension: 384
  - metric: "inner_product (faiss.IndexFlatIP)"
  - top_k: 5
  - query_instruction: false (raw query text without prefix)
Model IDs:
  - base_llm: PENDING
  - embedding_model: "BAAI/bge-small-en-v1.5"
Dataset:
  - split: smoke
  - version/hash: data/qa/retrieval_smoke.jsonl (15 grounded questions)
Seed: N/A
Metrics:
  - small (180/30 tokens, no instruction):
      recall@1: 0.800 (+0.067 vs EXP-001)
      recall@3: 1.000 (=)
      recall@5: 1.000 (=)
      mrr: 0.889 (+0.045)
      avg_latency_ms: 15.9 (-5.0 ms)
  - medium (300/50 tokens, no instruction):
      recall@1: 0.867 (+0.134 vs EXP-001)
      recall@3: 1.000 (+0.133 vs EXP-001)
      recall@5: 1.000 (=)
      mrr: 0.933 (+0.100 vs EXP-001)
      avg_latency_ms: 16.8 (-2.4 ms)
  - large (420/70 tokens, no instruction):
      recall@1: 0.733 (=)
      recall@3: 0.933 (+0.133 vs EXP-001)
      recall@5: 1.000 (=)
      mrr: 0.839 (+0.026 vs EXP-001)
      avg_latency_ms: 22.5 (-0.2 ms)
Conclusion: Disabling the generic BGE query instruction prefix markedly improves retrieval accuracy for domain-specific technical cybersecurity queries across all profiles. Medium profile without instruction achieved peak metrics across the study (86.7% Recall@1, 100% Recall@3 and Recall@5, 0.933 MRR, 16.8ms latency). Selected `medium` profile as production default in `config/rag.yaml`.
Artifact paths:
  - artifacts/faiss/medium
Notes:
  - Both modes remain supported via `config/rag.yaml` (`query_instruction_enabled: true/false`).
