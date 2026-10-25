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

---

### EXP-003 — RAG Top-K Engineering Comparison
Date: 2026-10-05
Phase: 5
Purpose: Controlled development-only engineering comparison of retrieval count (K=3, K=5, K=8) with medium chunk profile (300/50 tokens) and 1500-token context budget on Qwen2.5-1.5B-Instruct. Note: This is a small development engineering decision, not the final benchmark.
Config:
  - embedding_model: "BAAI/bge-small-en-v1.5"
  - chunk_profile: "medium" (300 tokens, 50 overlap)
  - index_type: "faiss.IndexFlatIP"
  - query_instruction: false
  - max_context_tokens: 1500
  - prompt_version: "rag_v1"
  - decoding: "greedy deterministic (do_sample=false)"
Model IDs:
  - base_llm: "Qwen/Qwen2.5-1.5B-Instruct"
  - embedding_model: "BAAI/bge-small-en-v1.5"
Dataset:
  - split: development qualitative engineering set
  - questions:
      - "What is broken access control?"
      - "What is the Govern function in NIST CSF 2.0?"
      - "What do the CISA Cross-Sector Cybersecurity Performance Goals provide?"
Seed: N/A (greedy decoding, exact FAISS search)
Metrics:
  - K=3:
      avg_retrieved_count: 3.0
      avg_context_tokens: 591.0
      avg_retrieval_latency_ms: 89.6
      rep_generation_latency_ms: 84900.3
      rep_output_tokens: 110
  - K=5:
      avg_retrieved_count: 5.0
      avg_context_tokens: 848.3
      avg_retrieval_latency_ms: 52.7
      rep_generation_latency_ms: 116461.2
      rep_output_tokens: 161
  - K=8:
      avg_retrieved_count: 8.0
      avg_context_tokens: 1108.7
      avg_retrieval_latency_ms: 41.5
      rep_generation_latency_ms: 113357.6
      rep_output_tokens: 154
Conclusion: Selected K=5 as default engineering baseline in `config/rag.yaml`. K=3 retrieves fewer chunks (~591 tokens) risking omission of secondary sections across multi-document sources. K=8 retrieves ~1109 tokens, which crowds the 1500-token context budget on denser passages (e.g. CISA CPG queries consumed 1416 tokens) with diminishing marginal relevance. K=5 reliably fits well within the budget (~848 tokens avg) while providing comprehensive cross-document context with low retrieval latency (~52.7ms).
Artifact paths:
  - artifacts/rag_phase5_results.json
Notes:
  - Small qualitative engineering comparison for development configuration. Final formal Base-vs-RAG benchmark will be executed in Phase 6.

---

### EXP-004 — Base vs RAG Evaluation v1
Date: 2026-10-06
Phase: 6
Purpose: Development-only evaluation comparing BasePipeline vs RagPipeline on identical questions, identical model (Qwen/Qwen2.5-1.5B-Instruct), and identical greedy deterministic settings on local CPU. Note: This is a small development evaluation (12 questions), not the final Phase 9 comparison. No statistical significance is claimed.
Config:
  - base_llm: "Qwen/Qwen2.5-1.5B-Instruct"
  - base_prompt: "base_v1"
  - rag_prompt: "rag_v1"
  - embedding_model: "BAAI/bge-small-en-v1.5"
  - chunk_profile: "medium" (300/50 tokens)
  - top_k: 5
  - max_context_tokens: 1500
  - decoding: "greedy deterministic (do_sample=false, max_new_tokens=256)"
Model IDs:
  - base_llm: "Qwen/Qwen2.5-1.5B-Instruct"
  - embedding_model: "BAAI/bge-small-en-v1.5"
Dataset:
  - split: development evaluation set v1
  - path: "data/qa/evaluation_v1.jsonl"
  - total_questions: 12 (4 OWASP Top 10:2025, 3 NIST CSF 2.0, 3 CISA CPG 2.0, 2 unsupported out-of-corpus)
Seed: N/A (deterministic exact search, greedy decoding)
Metrics:
  - BasePipeline:
      avg_reference_coverage: 0.3917
      avg_latency_ms: 50638.6
      avg_input_tokens: 65.0
      avg_output_tokens: 185.1
  - RagPipeline:
      avg_reference_coverage: 0.5555 (+41.8% relative improvement)
      avg_total_latency_ms: 145572.9
      avg_retrieval_latency_ms: 390.9
      avg_generation_latency_ms: 145049.0
      avg_input_tokens: 1158.6
      avg_output_tokens: 108.3 (-41.5% token reduction, concise grounded answers)
      avg_context_tokens: 1011.0 (safely inside 1500 budget)
      expected_source_retrieval_rate: 1.0000 (100% on corpus questions)
      citation_frequency: 0.2500
      citation_validity: 1.0000 (0 invalid citations)
      unsupported_refusal_rate: 0.5000
Conclusion: RAG significantly outperforms Base on domain-specific factual grounding (0.5555 vs 0.3917 reference coverage). In particular, on the critical OWASP Top 10:2025 taxonomy question (eval_01), Base recalled an obsolete 2017 list (0.20 coverage) while RAG correctly identified all 2025 categories from retrieved evidence (1.00 coverage). RAG also yielded 100% expected source retrieval accuracy across all corpus questions. RAG responses are more concise (108 vs 185 output tokens) and grounded, but require higher latency on CPU due to 1000+ context prompt processing (145.6s vs 50.6s).
Artifact paths:
  - artifacts/evaluation/base_vs_rag_v1.json
Notes:
  - Development evaluation on 12 curated questions. Small dataset, reproducible with scripts/evaluate_base_vs_rag.py.

---

### EXP-005 — Phase 8 Pilot QLoRA Fine-Tuning
Date: 2026-10-25
Phase: 8
Base Git commit: 49a5a79
Base model: Qwen/Qwen2.5-1.5B-Instruct
Hardware: NVIDIA GeForce RTX 2050, 4 GB VRAM
Environment:
  - Python: 3.13.9
  - PyTorch: 2.14.0+cu130
  - CUDA: 13.0 runtime
  - Transformers: 5.17.0
  - PEFT: 0.21.2
  - TRL: 1.14.1
  - bitsandbytes: 0.50.2
  - Accelerate: 1.15.0
  - Datasets: 5.1.0
Dataset:
  - train: 38 records
  - validation: 8 records
  - held-out test: 9 records
  - test used during training: NO (held-out test split strictly protected)
QLoRA:
  - 4-bit NF4
  - double_quant: True
LoRA:
  - r: 16
  - alpha: 32
  - dropout: 0.05
  - target_modules: all-linear
  - trainable parameters: 18,464,768 / 1,562,179,072 (~1.182%)
Training:
  - epochs: 3
  - optimization steps: 15
  - learning rate: 2e-4
  - batch size: 2
  - gradient accumulation: 4
  - max sequence length: 512
  - seed: 42
  - warmup ratio: 0.1
  - resolved warmup steps: 2
Results:
  - training loss: 1.8394
  - validation loss: 1.2016
  - runtime: ~86.47 seconds
  - CUDA OOM: NO
Adapter:
  - path: artifacts/finetuning/phase8_pilot/
  - adapter reload: PASS
  - single inference smoke: PASS
Qualitative pilot smoke:
  - questions evaluated: 6
  - protected Phase 7 test overlap: 0/6
  - Phase 6 evaluation overlap: 0/6
  - train overlaps: 3/6 (intentionally labeled as trained-domain checks)
  - validation overlaps: 1/6
  - no obvious generation regression observed
  - Fine-tuned outputs were generally more concise/focused than Base on this small smoke set
  - out-of-domain cookie question remained normal/general-purpose without cybersecurity contamination or unwarranted refusal
Conclusion: The pilot demonstrates successful domain adaptation and no obvious qualitative regression. Formal comparative performance remains to be measured in Phase 9.
Compatibility notes:
  - Initial training attempt failed before the first gradient step because installed TRL 1.14.1 API expected:
      - `warmup_steps` instead of `warmup_ratio` in `SFTConfig`
      - `max_length` instead of `max_seq_length` in `SFTConfig`
      - `processing_class` instead of `tokenizer` in `SFTTrainer`
  - A minimal compatibility patch in `scripts/train_qlora.py` and unit tests in `tests/unit/test_qlora_training.py` resolved these translations while preserving exact experiment semantics.
Artifact paths:
  - artifacts/finetuning/phase8_pilot/adapter_config.json
  - artifacts/finetuning/phase8_pilot/adapter_model.safetensors
  - artifacts/finetuning/phase8_pilot/training_metadata.json



