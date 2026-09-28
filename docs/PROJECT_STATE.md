# Project State

> **Rule:** The coding agent MUST read this file before making any change.
> Update this file at the end of every meaningful task.

---

## Current phase: 2 — Document Ingestion

**Status:** COMPLETE

---

## Phase 0 — Repository, Environment, and Guardrails

**Status:** COMPLETE (see git commit 55be644)

All Phase 0 acceptance criteria passed and verified. Full record in commit history.

---

## Phase 1 — Domain and Dataset Contract

**Status:** COMPLETE (see git commit ba43948)

All Phase 1 acceptance criteria passed and verified. Full record in commit history.

---

## Phase 2 — Document Ingestion

**Status:** COMPLETE

### Completed

- [x] Official NIST CSF 2.0 PDF downloaded from primary source (`https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf`), verified via PyMuPDF (32 pages, 68,335 chars), SHA-256 computed.
- [x] Official CISA CPG 2.0 PDF downloaded from primary source (`https://www.cisa.gov/sites/default/files/2025-12/CPG_Report_2.0_508c.pdf`), verified via PyMuPDF (36 pages, 95,324 chars), SHA-256 computed.
- [x] Canonical OWASP Top 10:2025 corpus acquired from official GitHub repository (`OWASP/Top10/2025/docs/en/*.md`, 16 Markdown files), individual SHA-256 hashes computed.
- [x] OWASP slide deck PDF discarded from canonical corpus role (retaining official Markdown documentation as canonical textual source).
- [x] Conservative text cleaner implemented (`src/contextforge/ingestion/cleaner.py`):
  - Preserves punctuation, casing, code blocks, URLs, and cybersecurity tokens (`CVE-2025-...`, `TLS 1.3`, `HTTP/2`, `XSS`, `SQL`, `< >`, `/`, `_`, `-`, `:`).
  - Normalizes via Unicode NFKC (handles ligatures like fi, fl).
  - Normalizes CRLF/CR to LF, strips non-printable ASCII control characters.
- [x] PDF loader implemented (`src/contextforge/ingestion/pdf_loader.py`):
  - PyMuPDF extraction with page-level provenance (1-indexed page numbers, char counts).
  - Scanned / image-only / empty PDF detection raising `NoExtractableTextError` (no silent OCR).
  - Safe file closure via `try...finally: doc.close()`.
- [x] Markdown loader implemented (`src/contextforge/ingestion/markdown_loader.py`):
  - Preserves headings and section-level provenance.
  - Full UTF-8 support.
- [x] Common types and error hierarchy implemented (`src/contextforge/common/types.py`):
  - `Document`, `ExtractedUnit`, `ExtractedDocument`.
  - `DocumentNotFoundError`, `UnsupportedDocumentTypeError`, `DocumentExtractionError`, `NoExtractableTextError`, `DuplicateDocumentError`.
- [x] SQLite storage layer implemented (`src/contextforge/storage/`):
  - `schema.sql`: `documents` table with `id`, `title`, `source_path`, `checksum` (UNIQUE), `source_type`, `status`, `created_at`, `metadata_json`.
  - `db.py`: Connection manager with WAL mode and foreign keys enabled.
  - `repositories.py`: Parameterized CRUD operations on `documents` table.
- [x] Ingestion service implemented (`src/contextforge/ingestion/service.py`):
  - End-to-end pipeline: path validation -> type detection -> SHA-256 -> duplicate check -> extraction -> conservative clean -> SQLite persistence.
  - Duplicate detection by SHA-256: returns `ALREADY_INGESTED` and skips insert (never creates duplicate rows).
- [x] Ingestion CLI implemented (`scripts/ingest.py`):
  - Supports single file and directory ingestion.
  - Structured output (`Status`, `Document ID`, `Title`, `Source type`, `SHA-256`, `Pages/sections`, `Characters`, `Database`).
  - Clean exit codes (0 for success/duplicate skipped, 1 for failure).
- [x] Real corpus ingested into `data/contextforge.db`:
  - NIST CSF 2.0 (1 PDF document, 32 pages)
  - CISA CPG 2.0 (1 PDF document, 36 pages)
  - OWASP Top 10:2025 (16 Markdown documents, 151 sections)
  - Total in DB: 18 documents.
- [x] Idempotency / duplicate prevention verified on real corpus (re-running ingestion outputs `ALREADY_INGESTED`, DB count remains 18).
- [x] Test suite: 32 tests passing (unit tests for cleaner, PDF loader, Markdown loader, hashing, storage, service, errors; integration test for CLI).
- [x] Manifest updated (`data/manifests/corpus_manifest.csv`) with 18 records, real dates, real paths, real SHA-256 hashes, zero fabricated data.
- [x] Architectural decisions recorded in `docs/DECISIONS.md` (D-011 through D-014).

---

## Active task

Phase 2 complete — awaiting human approval to begin Phase 3.

---

## Blockers

None.

---

## Last verified commands

```text
.venv\Scripts\python.exe -m pytest -v
  -> 32 passed in 2.28s

.venv\Scripts\python.exe scripts/ingest.py data/raw/nist-csf-2.0.pdf
  -> Status: INGESTED (32 pages, 68335 chars)

.venv\Scripts\python.exe scripts/ingest.py data/raw/nist-csf-2.0.pdf
  -> Status: ALREADY_INGESTED (skipped, DB count unchanged)

.venv\Scripts\python.exe scripts/ingest.py data/raw/cisa-cpg-2.0.pdf
  -> Status: INGESTED (36 pages, 95324 chars)

.venv\Scripts\python.exe scripts/ingest.py data/raw/owasp-top10-2025
  -> Ingestion batch complete: 16 newly ingested, 0 duplicate(s) skipped.

.venv\Scripts\python.exe scripts/ingest.py data/raw/owasp-top10-2025
  -> Ingestion batch complete: 0 newly ingested, 16 duplicate(s) skipped.

.venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('data/contextforge.db'); print(conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0])"
  -> 18
```

---

## Known issues / limitations

- No OCR dependency installed; scanned PDFs raise `NoExtractableTextError` by design.
- Python 3.13 runtime remains verified for PyMuPDF and SQLite.

---

## Phase 2 acceptance criteria

| Criterion | Status |
|---|---|
| Text can be extracted from normal text PDF (PyMuPDF) | VERIFIED |
| Page/section provenance survives extraction | VERIFIED |
| Duplicate file detected via SHA-256 and skipped | VERIFIED |
| Missing/corrupt file fails with clear explicit error | VERIFIED |
| No OCR dependency silently introduced | VERIFIED |
| Conservative text cleaning preserves technical tokens | VERIFIED |
| Markdown loader preserves headings and UTF-8 | VERIFIED |
| SQLite documents table populated with parameterized queries | VERIFIED |
| CLI `scripts/ingest.py` functional with structured output | VERIFIED |
| Unit and integration tests pass (32/32) | VERIFIED |
| Corpus manifest updated with true hashes and paths | VERIFIED |
| Zero Phase 3 components introduced | VERIFIED |

---

## Next allowed task

> **Phase 3 — Chunking, Embeddings, and FAISS**
>
> Gate: Human confirms Phase 2 completion and approves start of Phase 3.
> Tasks: Experiment with chunk size / overlap variants, integrate Sentence Transformers, build FAISS index.
> Do NOT begin Phase 3 without explicit approval.
