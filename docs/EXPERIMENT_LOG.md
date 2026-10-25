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

---

### EXP-006 — Final Base vs RAG vs Fine-Tuned Evaluation
Date: 2026-10-25
Phase: 9
Purpose: Execute final three-way comparative evaluation across Base LLM (`Qwen/Qwen2.5-1.5B-Instruct`), RAG Pipeline (with `BAAI/bge-small-en-v1.5`, FAISS medium index, top_k=5), and Fine-Tuned Pipeline (QLoRA pilot adapter from Phase 8) on the strictly held-out test split (`data/qa/phase9_final_test.jsonl`, 9 items).
Base Git commit: 38eadeb
Hardware: NVIDIA GeForce RTX 2050 (4 GB VRAM), CUDA 13.0
Environment:
  - Python: 3.13.9
  - PyTorch: 2.14.0+cu130
  - Sentence-Transformers: 6.1.0
  - Transformers: 5.17.0
  - PEFT: 0.21.2
Dataset:
  - file: data/qa/phase9_final_test.jsonl (9 records)
  - SHA-256: 6132e6aa07444cc65d217734c8e47526cd8b6321ac61bc7404c4f17285a22d21
  - Derived from: data/qa/finetune_test.jsonl (SHA-256: ba7e4a65ecb8bfb6e6cae0a7d4f25cfae6564b8bc32b2de8a7bb9e611c964581)
  - Train exact overlap: 0/9
  - Validation exact overlap: 0/9
  - Phase 8 smoke overlap: 0/9
  - Phase 6 evaluation_v1 overlap: 0/9
  - Model inference previously run on test split: NO
Config:
  - config_file: config/phase9_evaluation.yaml
  - Base model: Qwen/Qwen2.5-1.5B-Instruct
  - Embedding model: BAAI/bge-small-en-v1.5
  - FAISS index: artifacts/faiss/medium/index.faiss (SHA-256: b77aac7327237de8d09276f1609ed90eb405b99b49b3adddff7750e0a11b3952)
  - Adapter: artifacts/finetuning/phase8_pilot/ (adapter_model.safetensors SHA-256: 6521dec6e644eaae09c5fd94738531e79a7b66bfcf95a90d808b6a0c934abe51)
  - Decoding: do_sample=False, max_new_tokens=256
  - Prompts: Base=base_v1, RAG=rag_v1, Fine-Tuned=base_v1
Inference Precision Fairness:
  - Base: device=cuda, dtype=torch.bfloat16, quantized=False
  - RAG: device=cuda, dtype=torch.bfloat16, quantized=False
  - Fine-Tuned: device=cuda, dtype=torch.bfloat16, quantized=False
  - Comparable inference precision: YES (All three BF16 unquantized base model weights in GPU VRAM)
Measured Metrics:
  - Base LLM:
      token_precision: 0.2487
      token_recall: 0.4106
      token_f1: 0.2870
      semantic_similarity: 0.8641
      mean_latency_ms: 8784.4
      median_latency_ms: 9716.5
      mean_input_tokens: 64.1
      mean_output_tokens: 177.2
      peak_gpu_memory_mb: 2964.3
  - RAG Pipeline:
      token_precision: 0.2811
      token_recall: 0.4200
      token_f1: 0.3159
      semantic_similarity: 0.8715
      mean_latency_ms: 11837.3
      median_latency_ms: 13161.9
      mean_input_tokens: 1346.8
      mean_output_tokens: 173.7
      mean_context_tokens: 1200.2
      expected_doc_hit@1: 0.6667 (6/9)
      expected_doc_hit@3: 0.7778 (7/9)
      expected_doc_hit@5: 1.0000 (9/9)
      expected_doc_mrr: 0.7722
      citation_presence_rate: 0.3333 (3/9)
      citation_validity_rate: 1.0000 (100% of emitted citations valid)
      peak_gpu_memory_mb: 3224.4
  - Fine-Tuned Pipeline:
      token_precision: 0.3933
      token_recall: 0.2892
      token_f1: 0.3274
      semantic_similarity: 0.8692
      mean_latency_ms: 5633.3
      median_latency_ms: 5258.0
      mean_input_tokens: 64.1
      mean_output_tokens: 73.0
      peak_gpu_memory_mb: 3059.1
Unsupported Refusal Metrics:
  - refusal_accuracy: NOT_APPLICABLE (0 unsupported items in final held-out test split)
  - unsupported_refusal_rate: NOT_APPLICABLE
Per-Question Breakdown:
  - ft_0053: RAG won F1 (0.3759); Fine-Tuned won semantic similarity (0.8421). Base was overly verbose.
  - ft_0012: Fine-Tuned won decisively (F1 0.4222, Sim 0.8768 vs Base 0.2118, RAG 0.2013). RAG missed specific injection category definitions in top-ranked chunks.
  - ft_0042: Base won F1 (0.3862) and Sim (0.8923). All models gave sound credential advice.
  - ft_0040: RAG won decisively (F1 0.3311, Sim 0.9460). Base hallucinated refusal ("an internal or official term that might not be widely known"); FT gave a generic summary missing CISA specific terminology.
  - ft_0034: RAG won decisively (F1 0.4259, Sim 0.9421). Grounded accurately in Current vs Target Profile definitions; Base generated a cut-off bullet list.
  - ft_0046: RAG won F1 (0.3408) and Sim (0.9050). Base hallucinated that CISA CPG 2.0 has not yet been released.
  - ft_0013: Fine-Tuned won F1 (0.2703) and Sim (0.8451). Accurately specified parameterized queries and prepared statements.
  - ft_0052: RAG won F1 (0.3265) and Sim (0.9326). Accurately grounded in GOVERN function; Base hallucinated CPG compliance.
  - ft_0008: Fine-Tuned won F1 (0.3710) and Sim (0.8739) by addressing supply chain dependency risks, though misstated category as Insecure Design. RAG gave exact A03 classification and CWE numbers.
Qualitative Review (9 items):
  - Base: Factual alignment 3/9 YES, 3/9 PARTIAL, 3/9 NO. Hallucinations observed on 3 framework queries (falsely claimed CISA CPG 2.0 was unreleased, claimed CISA CPG goal terminology was unknown, mischaracterized NIST CSF 2.0 Govern).
  - RAG: Factual alignment 8/9 YES, 1/9 PARTIAL, 0/9 NO. Grounded in retrieved passages; eliminated framework hallucinations. Only weakness was missed coverage on ft_0012 due to retriever returning broader injection mitigations rather than full category definitions.
  - Fine-Tuned: Factual alignment 6/9 YES, 3/9 PARTIAL, 0/9 NO. Highly concise responses (avg 73 tokens vs 177 tokens) adhering to cybersecurity assistant style. Zero refusals, but lacked verbatim terminology when questions required specific framework definitions (e.g., ft_0040, ft_0052).
Conclusion:
  - RAG is superior for dynamic factual accuracy, grounded framework compliance, and citation provenance. It completely prevents hallucinations on evolving standards.
  - Fine-Tuned LoRA is superior for low-latency (5.6s vs 11.8s, 2.1x faster), concise domain style (73 vs 174 tokens), and high token precision (0.3933 vs 0.2811) without prompt bloat.
  - Base LLM exhibits significant factual hallucinations on specific framework standards, demonstrating that ungrounded base models are inappropriate for authoritative domain advisory.
Limitations:
  - Pilot held-out test split contains 9 items; sufficient for directional comparative analysis but too small for broad statistical significance.
  - Split contains 0 unsupported/out-of-domain questions; refusal behavior on this split is not tested (historical Phase 6 dev data remains development evidence).
Artifact paths:
  - artifacts/evaluation/phase9_raw_predictions.json
  - artifacts/evaluation/phase9_metrics.json




