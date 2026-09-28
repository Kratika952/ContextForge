# Dataset Contract

> **Purpose:** Define the structure, rules, and integrity constraints for all QA datasets used by ContextForge.
>
> **Scope:** Phase 1 — contracts and schemas only. No QA examples are populated in this phase.
>
> **Rule:** Any future QA example generation must comply with this contract. No code, prompt, or manual process may violate the leakage-prevention rules defined here.

---

## 1. QA Task Definition

**Task:** Evidence-grounded domain-specific question answering over defensive cybersecurity documentation.

A valid question:
- relates to a concept, control, framework element, risk, or mitigation present in the approved corpus
- has an answer that can be grounded in specific corpus text
- is answerable without requiring knowledge outside the approved corpus

A valid answer:
- directly addresses the question
- stays within the defensive cybersecurity scope
- can cite the specific document section(s) that support it
- acknowledges when the evidence is insufficient or ambiguous

---

## 2. Question Category Taxonomy

Use exactly these category labels in all QA records. Do not invent new categories without updating this document.

| Category | Definition | Synthetic examples |
|---|---|---|
| `factual_definition` | Questions asking what a term, concept, vulnerability class, or framework element *is* | "What is SQL injection?" / "What does the NIST CSF 'Identify' function cover?" |
| `recommendation` | Questions asking what is recommended as best practice in the corpus | "What does OWASP recommend to prevent broken access control?" |
| `control_guidance` | Questions asking how to implement a specific control or security measure | "How should multi-factor authentication be configured according to the CISA CPGs?" |
| `framework_mapping` | Questions asking how one framework element relates to or maps to another | "How do the CISA CPGs align with NIST CSF 2.0 functions?" |
| `risk_explanation` | Questions asking why a particular vulnerability or gap is risky | "Why is insecure design considered a top-10 web application risk?" |
| `mitigation` | Questions asking how to reduce or remediate a specific risk or vulnerability | "What mitigations does OWASP recommend for cryptographic failures?" |
| `procedural_guidance` | Questions asking about processes, workflows, or step-by-step approaches | "What steps does NIST CSF 2.0 describe for incident response?" |

> **Note on synthetic examples:** The examples above illustrate category meaning only. They must **not** be automatically inserted into the benchmark dataset. All benchmark examples must be manually reviewed before inclusion.

---

## 3. Answer Quality Contract

Every answer produced by any ContextForge inference path (base, RAG, fine-tuned) will eventually be evaluated against these dimensions.

### 3.1 Quality Dimensions

| Dimension | Definition |
|---|---|
| **Correctness** | The answer accurately reflects what the source documents state. It does not contradict the corpus. |
| **Relevance** | The answer addresses the actual question asked, not a related but different question. |
| **Groundedness** | Claims in the answer are supported by specific evidence from the corpus. Unsupported claims are explicitly flagged. |
| **Completeness** | The answer covers the main aspects of the question given the evidence available. Partial answers are permitted but must acknowledge the limitation. |
| **Citation/Provenance accuracy** | For RAG responses: cited sources are real, correspond to the retrieved chunks, and are not invented. For base/fine-tuned responses: no fabricated citations are acceptable. |

### 3.2 Required Answer Behaviours

A correct ContextForge answer MUST:
- directly address the user's question
- remain within the defensive cybersecurity scope of the approved corpus
- use corpus evidence to support claims (RAG path)
- distinguish when the available evidence is insufficient to answer confidently
- preserve uncertainty when the corpus text is ambiguous or silent
- provide real source provenance in RAG responses (document ID, section where applicable)

A correct ContextForge answer MUST NOT:
- invent standards, framework versions, or section numbers not present in the corpus
- fabricate citations or recommend non-existent controls
- produce offensive security guidance even if asked
- conflate different framework versions without noting the distinction

### 3.3 Scoring

Numeric scoring weights and thresholds are **not assigned in Phase 1**. They will be determined during Phase 6 (Evaluation Harness) based on the evaluation dataset and pilot runs.

---

## 4. Dataset Splits

### 4.1 Split Definitions

| Split | File | Purpose |
|---|---|---|
| Training | `data/qa/train.jsonl` | QLoRA fine-tuning (Phase 8) |
| Validation | `data/qa/validation.jsonl` | Prompt selection, chunking experiments, retrieval top-k experiments, fine-tuning hyperparameter decisions |
| Test | `data/qa/test.jsonl` | **Final evaluation only** — never used for any tuning decision |

### 4.2 Split Rules

#### Training set
- Used exclusively for QLoRA fine-tuning.
- May contain question–answer pairs generated from corpus content.
- Must not contain any question–answer pair that is also in the test set.
- Must not contain deliberate paraphrases of test questions.

#### Validation set
- Used for all intermediate decisions: prompt wording, chunk size selection, top-k selection, embedding model selection, and fine-tuning hyperparameter tuning.
- Must not contain any question–answer pair that is also in the test set.
- A decision recorded as "tuned on validation" must include a reference to the validation experiment in `docs/EXPERIMENT_LOG.md`.

#### Test set
- Used **only** for the final three-way comparative evaluation (Phase 9).
- Frozen once final evaluation begins. No item may be added, removed, or edited after freezing.
- **Absolutely prohibited uses of the test set:**
  - Training any model
  - Prompt tuning or prompt selection
  - Chunk size selection
  - Top-k selection
  - Embedding model selection
  - Hyperparameter tuning
  - Model selection
  - Any iterative improvement loop

> The test set is sacred. Any accidental use of test data for optimization must be documented in `docs/EXPERIMENT_LOG.md` as a data-integrity incident and the affected results must be clearly flagged as invalid.

---

## 5. Leakage Prevention Rules

### 5.1 Key Distinction

> **Document corpus overlap ≠ QA-example leakage.**

All three inference paths (base, RAG, fine-tuned) share the same document corpus. It is expected and acceptable that:
- the RAG index contains the same documents used to write training examples
- the fine-tuned model was trained on text derived from the same corpus

This is **not** leakage.

Leakage occurs when a **specific QA question or its answer** from the test set is used to:
- create a training example
- select a system configuration
- tune a prompt

### 5.2 Leakage Prevention Procedures

1. **Stable IDs:** Every QA example has a unique `id` (e.g., `train_0001`, `val_0001`, `test_0001`). IDs are assigned at creation and never reused across splits.

2. **Split-before-generate:** The split assignment (`train`, `validation`, `test`) must be made **before** any examples are generated. Do not generate examples and then split; split first, then fill each split independently.

3. **Duplicate detection:** Before committing any QA dataset, run exact-string and fuzzy-match deduplication across all three splits. Remove or quarantine near-duplicates that span splits.

4. **No paraphrase seeding:** Do not paraphrase a test question to create a training example. This is leakage.

5. **Corpus overlap is allowed:** The same source document may contribute examples to all three splits. The distinction is whether the *specific question-answer pair* crosses the split boundary, not whether the underlying document does.

6. **Reviewer sign-off:** Before the test set is frozen, a human must review a sample of the test questions to confirm they have not appeared in training or validation.

7. **Freeze record:** When the test set is frozen, record the freeze event (date, question count, dataset hash) in `docs/EXPERIMENT_LOG.md`.

8. **Post-freeze immutability:** Once frozen, the test set must not be modified. If an error is discovered in a test item after freezing, document it but do not silently edit the item; instead, flag that item's results as suspect.

---

## 6. QA Record Schemas

### 6.1 Training record (`data/qa/train.jsonl`)

```json
{
  "id": "train_0001",
  "question": "...",
  "answer": "...",
  "category": "factual_definition | recommendation | control_guidance | framework_mapping | risk_explanation | mitigation | procedural_guidance",
  "source_document_ids": ["owasp-top10-2025"],
  "metadata": {
    "created_at": "YYYY-MM-DD",
    "annotator": "human | semi-auto-reviewed",
    "notes": ""
  }
}
```

### 6.2 Validation record (`data/qa/validation.jsonl`)

```json
{
  "id": "val_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "relevant_document_ids": ["nist-csf-2.0"],
  "metadata": {
    "created_at": "YYYY-MM-DD",
    "annotator": "human | semi-auto-reviewed",
    "notes": ""
  }
}
```

### 6.3 Test record (`data/qa/test.jsonl`)

```json
{
  "id": "test_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "relevant_document_ids": ["cisa-cpg-2.0"],
  "metadata": {
    "created_at": "YYYY-MM-DD",
    "annotator": "human | semi-auto-reviewed",
    "frozen_at": "YYYY-MM-DD",
    "notes": ""
  }
}
```

### 6.4 Schema notes

- `id` is globally unique and stable. Never reuse an ID.
- `category` must be one of the seven values defined in §2.
- `source_document_ids` / `relevant_document_ids` must reference `document_id` values in `data/manifests/corpus_manifest.csv`.
- `reference_answer` is the ground-truth answer for evaluation. It must be grounded in the corpus.
- `annotator` distinguishes fully human-written examples from semi-automatically generated examples that have been human-reviewed.
- `metadata` is extensible but must never include fields that could leak split identity to training code.

---

## 7. Minimum Evaluation Dataset Sizes

These are targets, not hard limits. Final numbers will be determined during Phase 6 and Phase 7.

| Split | Minimum target | Notes |
|---|---|---|
| Training | 200 examples | Enough for a small QLoRA run; more is better |
| Validation | 50 examples | Enough for prompt and config experiments |
| Test | 50 examples | Enough for statistically meaningful three-way comparison |

---

## 8. Research Questions

The following research questions guide the evaluation design. These are questions to be investigated, not predetermined conclusions.

1. How much does RAG improve answer correctness over the base model for cybersecurity-domain QA?
2. How does QLoRA domain fine-tuning compare with RAG on answer quality, groundedness, and completeness?
3. Which inference path produces fewer unsupported or fabricated claims?
4. What is the retrieval quality (Recall@k, MRR) of the FAISS-based RAG system on the cybersecurity corpus?
5. What are the latency and GPU-memory tradeoffs of each inference path?
6. For which question categories (factual definition, framework mapping, procedural guidance, etc.) does each approach perform best and worst?
7. Is the fine-tuned model sufficiently calibrated to acknowledge insufficient evidence, or does it over-confidently answer out-of-corpus questions?
