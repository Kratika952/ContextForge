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
