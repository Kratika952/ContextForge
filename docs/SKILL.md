---
name: domainadapt-project-guardrail
description: Operating rules for coding agents working on the DomainAdapt NLP project. Enforces phased implementation, repository inspection, reproducible experiments, fixed architecture boundaries, testing, and anti-hallucination behavior.
---

# DomainAdapt Coding Skill

## Purpose

Use this skill whenever modifying, debugging, planning, testing, or reviewing the DomainAdapt repository.

This skill does **not** contain the whole project specification. The canonical implementation plan is:

```text
docs/IMPLEMENTATION_PLAN.md
```

The current execution state is:

```text
docs/PROJECT_STATE.md
```

Project decisions are recorded in:

```text
docs/DECISIONS.md
```

Experiment history is recorded in:

```text
docs/EXPERIMENT_LOG.md
```

---

# Mandatory First Step

Before taking any action:

1. Inspect the repository.
2. Read `docs/IMPLEMENTATION_PLAN.md`.
3. Read `docs/PROJECT_STATE.md`.
4. Read relevant entries in `docs/DECISIONS.md`.
5. Inspect the files relevant to the requested task.
6. Run or inspect the current test state if the task affects executable code.

Never assume those files contain what you remember from an earlier session.

If one of the mandatory project docs does not exist, treat that as a Phase 0 issue rather than inventing its contents.

---

# Source-of-Truth Priority

When instructions conflict, use this project-level order:

1. Human's explicit current request
2. `docs/DECISIONS.md` for already approved project choices
3. `docs/IMPLEMENTATION_PLAN.md`
4. `docs/PROJECT_STATE.md` for current phase/status
5. Existing tested code behavior
6. README or older notes

Do not silently resolve a meaningful conflict. Report it.

---

# Core Project Contract

DomainAdapt compares exactly three paths:

```text
1. Base LLM
2. Same Base LLM + RAG
3. Same Base LLM + LoRA/QLoRA adapter
```

For a fair experiment:

- Base and RAG use the same base model.
- Fine-tuned uses that same base model plus adapter.
- RAG uses Sentence Transformers + FAISS.
- Normal application metadata/results use SQLite.
- Backend uses FastAPI.
- Frontend uses Streamlit.
- Core ML uses PyTorch + Hugging Face Transformers.
- Fine-tuning uses PEFT + TRL.
- Evaluation is reproducible and offline-first.

Do not replace this architecture without explicit human approval.

---

# Technologies You Must Not Introduce Without Approval

Do not add:

- LangChain
- LlamaIndex
- PostgreSQL
- MongoDB
- Redis
- Celery
- Kafka
- agent frameworks
- cloud-managed databases
- paid LLM APIs
- Docker as a mandatory dependency
- Kubernetes
- microservices

A library may be added only if:

1. the active phase needs it,
2. no existing dependency solves the requirement cleanly,
3. the reason is documented,
4. the human approves it when it changes the agreed stack.

---

# Phase Discipline

Work on one phase at a time.

Before coding, identify:

```text
Current phase:
Requested task:
Acceptance criterion being addressed:
Files to inspect:
Likely files to modify:
Tests to run:
```

Do not perform work belonging to a later phase merely because it seems convenient.

Examples:

- Do not build Streamlit during ingestion.
- Do not fine-tune before the evaluation harness exists.
- Do not add authentication during the MVP.
- Do not optimize performance before measuring a bottleneck.

If the user explicitly requests a cross-phase task, perform it but document the exception in `PROJECT_STATE.md`.

---

# Decision Gates

Never silently choose:

- project domain
- corpus
- base LLM
- embedding model
- chunk size/overlap as a “final” value
- top-k as a “final” value
- training dataset format
- QLoRA hyperparameters as final
- evaluation thresholds

Use the relevant phase's experiment/decision process.

If a decision is missing and blocks correct implementation, stop at that gate and ask the human.

---

# Repository Inspection Rules

Never say:

- “the project already has…”
- “this file contains…”
- “the tests are…”
- “the schema uses…”

unless you inspected it in the current work session.

Before editing a file:

1. read it,
2. identify its current responsibility,
3. check imports/usages if changing a shared API,
4. preserve unrelated behavior.

Never overwrite a file from a guessed template if it already exists.

---

# Minimal-Change Rule

Prefer the smallest change that satisfies the current acceptance criterion.

Avoid:

- sweeping refactors
- renaming unrelated modules
- creating generic “manager/factory/service” layers without a demonstrated need
- adding dependencies for one trivial helper
- rewriting working code because of stylistic preference

Refactor only when:

- required to satisfy current architecture,
- required to fix a verified defect,
- or explicitly requested.

---

# Common Interface Rule

All inference pipelines must return the shared `AnswerResult` concept.

Conceptually:

```python
AnswerResult:
    question
    pipeline
    answer
    model_id
    latency_ms
    sources
    metadata
```

Do not invent a different output shape for every pipeline.

RAG uses `sources`.
Base and Fine-tuned normally return an empty source list.

---

# RAG Rules

RAG flow must remain:

```text
question
-> embedding
-> FAISS retrieval
-> chunk lookup/provenance
-> context formatting
-> versioned prompt
-> SAME base LLM
-> answer + application-generated source metadata
```

Rules:

- FAISS is not the application database.
- SQLite stores structured metadata.
- Retrieved vector IDs must resolve to real chunk IDs.
- Source labels are generated by code, not invented by the LLM.
- Do not silently fall back to base inference if retrieval fails.
- Insufficient evidence must be representable.
- Do not dump the full corpus into context.

---

# Fine-Tuning Rules

QLoRA/LoRA training must:

- use the approved base model
- train only on the train split
- tune decisions against validation
- never use final test answers for training/tuning
- save training config
- save random seed
- save adapter artifacts
- record failed runs
- keep base-model identity attached to the adapter metadata

Do not claim “fine-tuning completed” unless a real training run completed and artifacts were produced.

---

# Evaluation Rules

The held-out test set is sacred.

Never:

- edit test questions because the model performs badly
- add test answers to training
- tune prompts on test outcomes
- select chunking/top-k from test metrics
- report validation metrics as final test results
- invent benchmark numbers

Save raw predictions before summary calculations.

Separate metrics into:

1. retrieval quality
2. answer quality
3. groundedness/support
4. latency/resources

Do not treat one automated “hallucination score” as ground truth.

If using LLM-as-a-judge, label it clearly and keep it separate from human evaluation.

---

# Testing Rules

For every code change:

1. add or update the narrowest relevant test,
2. run that test,
3. run the affected test group,
4. run the full suite when practical before phase completion.

Never claim tests passed unless they were actually run.

If a test cannot be run because of missing GPU/model/network:

```text
STATUS: UNVERIFIED
REASON: ...
```

Then run the closest deterministic local tests.

Real-model smoke tests should be separated from lightweight unit tests.

---

# Dependency/API Uncertainty Rule

If uncertain about a third-party library API:

- inspect the installed version/docs/source when available,
- or verify with a minimal executable experiment.

Do not invent function names, class arguments, return types, or package behavior from vague memory.

When compatibility matters, record the version used.

---

# Error Handling Rules

Never hide component failures.

Bad:

```python
try:
    ...
except Exception:
    return ""
```

Required behavior:

- raise/return a clear domain-specific error,
- preserve useful context,
- log enough to debug,
- do not expose secrets.

Explicitly handle major cases:

- missing/corrupt PDF
- scanned PDF with no text
- empty query
- missing FAISS index
- embedding/index mismatch
- missing chunk metadata
- model load failure
- incompatible adapter
- CUDA OOM
- malformed dataset
- DB errors

---

# Data and Provenance Rules

Every ingested document should have stable identity/checksum.

Every chunk should preserve:

- document ID
- chunk index
- page provenance when possible
- text
- content hash

Every training example should preserve source/provenance when practical.

Every final evaluation item should have stable ID.

Do not fabricate provenance.

---

# SQLite Rules

Use parameterized queries.

Do not interpolate user text into SQL strings.

Keep migrations/schema changes deliberate.

Before changing a frozen table shape:

1. inspect current data model,
2. identify code impacted,
3. propose migration/update,
4. update tests,
5. update implementation/decision docs if architecture changes.

---

# Secrets and Privacy

Never commit:

- API keys
- tokens
- private credentials
- `.env`
- private corpus data unless explicitly approved

Use `.env.example` with placeholder variable names only.

Do not log full private documents unnecessarily.

---

# Cost Rule

The baseline project must not require a paid API.

Prefer:

- local CPU for preprocessing/testing
- local or free GPU for inference where possible
- Kaggle/Colab for training
- persisted adapters so training is not repeated unnecessarily

Do not start a paid resource or add a paid dependency without human approval.

---

# Documentation Rule

When behavior or architecture changes, update the relevant document in the same task.

At minimum, consider:

- `docs/PROJECT_STATE.md`
- `docs/DECISIONS.md`
- `docs/EXPERIMENT_LOG.md`
- README
- config examples

Do not let documentation describe code that does not exist yet as if it is implemented.

Use explicit statuses such as:

```text
PLANNED
IN_PROGRESS
IMPLEMENTED
VERIFIED
UNVERIFIED
```

---

# PROJECT_STATE Update Rule

At the end of every meaningful task, update:

```markdown
Current phase:
Status:

Completed:
- ...

Active task:
...

Verified commands:
- ...

Known issues:
- ...

Next allowed task:
...
```

A phase may become `COMPLETE` only when every phase acceptance criterion from `IMPLEMENTATION_PLAN.md` is satisfied.

---

# Experiment Logging Rule

Any change intended to compare quality/performance must receive an experiment record.

Record:

- purpose
- config
- model IDs
- dataset hash/version
- seed
- metrics
- conclusion
- artifact paths

Do not overwrite old results.

---

# Before-Action Checklist

Before writing code, verify:

- [ ] I read the implementation plan.
- [ ] I read project state.
- [ ] I know the active phase.
- [ ] I inspected the relevant existing files.
- [ ] I am addressing a specific acceptance criterion.
- [ ] I am not introducing an unapproved technology.
- [ ] I know what test will prove the change works.
- [ ] I am not using the held-out test set for tuning.

If any required box is false, resolve it before implementation.

---

# After-Action Checklist

Before declaring completion:

- [ ] Code change is minimal and phase-appropriate.
- [ ] Tests were added/updated.
- [ ] Relevant tests were actually run.
- [ ] Results are reported accurately.
- [ ] No benchmark numbers were invented.
- [ ] No architecture change was hidden.
- [ ] Docs/config were updated where needed.
- [ ] `PROJECT_STATE.md` was updated.
- [ ] Remaining limitations are stated.
- [ ] Next allowed task is clear.

---

# Required Work Summary Format

After a coding task, report:

```text
Phase:
Task:

Changed:
- file
- file

Verified:
- command -> result
- command -> result

Acceptance criteria:
- PASS/FAIL ...

Unverified:
- ...

Known issues:
- ...

Project state:
- updated / not applicable

Next:
- ...
```

If something failed, say it failed. Do not soften it into “should work.”

---

# Stop Conditions

Stop and ask the human instead of guessing when:

- required domain is not chosen
- base/embedding model decision is required and no approved choice exists
- a requested change conflicts with the locked architecture
- data licensing/provenance is unclear
- fixing a task requires destructive data loss
- test-set integrity would be compromised
- a paid service would be required
- a secret/credential is required
- an existing behavior conflicts materially with the implementation plan

---

# Guiding Principle

> **Inspect first. Change one thing. Prove it works. Record the result. Then continue.**

The goal is not to produce the most code. The goal is to produce a correct, reproducible DomainAdapt system whose claims are supported by actual tests and experiments.
