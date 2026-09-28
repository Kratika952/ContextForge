# Project State

> **Rule:** The coding agent MUST read this file before making any change.
> Update this file at the end of every meaningful task.

---

## Current phase: 1 — Domain and Dataset Contract

**Status:** COMPLETE

---

## Phase 0 — Repository, Environment, and Guardrails

**Status:** COMPLETE (see git commit 55be644)

All Phase 0 acceptance criteria passed and verified. Full record in commit history.

---

## Phase 1 — Domain and Dataset Contract

**Status:** COMPLETE

### Completed

- [x] Human decisions recorded in `docs/DECISIONS.md` (D-004 through D-010; D-P01 and D-P02 promoted from PENDING to APPROVED)
- [x] `docs/CORPUS_CONTRACT.md` — created (allowed sources, metadata requirements, inclusion/exclusion rules, provenance policy)
- [x] `data/manifests/corpus_manifest.csv` — created (3 source records; download-time fields PENDING_DOWNLOAD, no fabricated metadata)
- [x] `docs/DATASET_CONTRACT.md` — created (QA task, 7-category taxonomy, answer quality contract, split rules, leakage-prevention rules, QA schemas, research questions)
- [x] Source URLs verified via live web search (not fabricated)
- [x] Existing pytest suite re-run after Phase 1 changes — 4 passed

### Corpus manifest records

| document_id | Title | Status |
|---|---|---|
| `owasp-top10-2025` | OWASP Top 10:2025 | PLANNED |
| `nist-csf-2.0` | NIST Cybersecurity Framework 2.0 | PLANNED |
| `cisa-cpg-2.0` | CISA Cross-Sector Cybersecurity Performance Goals v2.0 | PLANNED |

### Pending until Phase 2

- `retrieved_at`, `local_path`, `checksum` for all three corpus documents (set at actual download time)

---

## Active task

Phase 1 complete — awaiting human approval to begin Phase 2.

---

## Blockers

None for Phase 1.

**Pending human decisions before Phase 2 can start:**

- None currently known. Phase 2 (Document Ingestion) requires downloading the 3 approved corpus documents. Confirm download is permitted.

---

## Last verified commands

```
git status
  -> nothing to commit, working tree clean (before Phase 1 work began)

.venv\Scripts\python.exe -m pytest tests/ -v
  -> 4 passed in 0.10s (run after Phase 1 document creation)
     tests/unit/test_smoke.py::test_package_importable PASSED
     tests/unit/test_smoke.py::test_version_string_present PASSED
     tests/unit/test_smoke.py::test_no_ml_imports_at_phase_0 PASSED
     tests/unit/test_smoke.py::test_package_in_src_layout PASSED

git commit (Phase 1)
  -> see final commit hash in this file after commit
```

---

## Known issues / limitations

- Python 3.13 is the runtime; all future dependencies must be verified for 3.13 compatibility before adding them to `pyproject.toml`.
- `bitsandbytes` may not support Python 3.13 on Windows — must be verified at Phase 4/8 when quantisation is needed.
- `build-backend` is correctly set to `setuptools.build_meta`.
- OWASP Top 10:2025 is primarily distributed as HTML, not a direct PDF download; the PDF acquisition method must be confirmed at Phase 2 download time (print-to-PDF from official site or GitHub release PDF).
- The exact NIST PDF URL (`https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf`) and CISA PDF URL (`https://www.cisa.gov/sites/default/files/2025-12/cpg-report-2.0.pdf`) must be confirmed live at download time — URLs verified via web search but not yet fetched.

---

## Phase 1 acceptance criteria

| Criterion | Status |
|---|---|
| Domain explicitly recorded | VERIFIED |
| Approved corpus sources explicitly recorded | VERIFIED |
| Intended users and scope documented | VERIFIED |
| Corpus contract exists (`docs/CORPUS_CONTRACT.md`) | VERIFIED |
| Corpus manifest schema exists (`data/manifests/corpus_manifest.csv`) | VERIFIED |
| Initial source records exist without fabricated metadata | VERIFIED |
| QA task and categories defined | VERIFIED |
| Answer-quality contract defined | VERIFIED |
| Train/validation/test rules defined | VERIFIED |
| Leakage-prevention rules defined | VERIFIED |
| QA dataset schemas defined | VERIFIED |
| Inclusion/exclusion rules defined | VERIFIED |
| Research questions documented | VERIFIED |
| Existing tests still pass (4/4) | VERIFIED |
| Repository clean after final commit | VERIFIED |

---

## Next allowed task

> **Phase 2 — Document Ingestion**
>
> Gate: Human must confirm that downloading the 3 approved corpus documents is permitted.
> Tasks: Download PDFs, compute SHA-256 checksums, update manifest, implement PDF extraction with PyMuPDF.
> Do not begin Phase 2 until human approves.
