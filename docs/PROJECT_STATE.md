# Project State

> **Rule:** The coding agent MUST read this file before making any change.
> Update this file at the end of every meaningful task.

---

## Current phase: 0 — Repository, Environment, and Guardrails

**Status:** COMPLETE

---

## Completed

- [x] Git repository initialised (`git init`)
- [x] `docs/IMPLEMENTATION_PLAN.md` — present (pre-existing)
- [x] `docs/SKILL.md` — present (pre-existing)
- [x] `docs/PROJECT_STATE.md` — created (this file)
- [x] `docs/DECISIONS.md` — created
- [x] `docs/EXPERIMENT_LOG.md` — created
- [x] `.gitignore` — created (covers venvs, secrets, model caches, FAISS, adapters, SQLite, OS/IDE)
- [x] `.env.example` — created (placeholder variables only, no real credentials)
- [x] `pyproject.toml` — created (setuptools src-layout, pytest config, `contextforge` package)
- [x] `README.md` — created
- [x] `config/project.yaml` — created
- [x] `src/contextforge/__init__.py` — created (Phase 0 skeleton)
- [x] `tests/unit/test_smoke.py` — created (4 smoke tests)
- [x] Directory stubs: `data/raw/`, `data/processed/`, `data/qa/`, `data/manifests/`, `artifacts/faiss/`, `artifacts/adapters/`, `artifacts/evaluation/`, `artifacts/plots/`, `scripts/`, `frontend/`, `tests/integration/`, `tests/fixtures/`
- [x] Phase 0 final verification pass completed — full implementation plan read (all 1633 lines), all acceptance criteria re-verified, all tests re-run

---

## Active task

Phase 0 complete — awaiting human verification and Phase 1 approval.

---

## Blockers

None for Phase 0.

**Pending human decisions before Phase 1 can start:**

- Domain (HUMAN_DECISION_REQUIRED)
- Corpus sources (HUMAN_DECISION_REQUIRED)

---

## Last verified commands

```
git init
  -> Initialized empty Git repository in .../ContextForge/.git/

python --version
  -> Python 3.13.9

pip --version
  -> pip 25.2

.venv\Scripts\python.exe -m pip install -e ".[dev]"
  -> Successfully installed contextforge-0.1.0 pytest-9.1.1 pytest-cov-7.1.0 (EXIT: 0)

.venv\Scripts\python.exe -c "import contextforge; print(contextforge.__version__)"
  -> import OK, version: 0.1.0

.venv\Scripts\python.exe -m pytest tests/ -v
  -> 4 passed in 0.15s
     tests/unit/test_smoke.py::test_package_importable PASSED
     tests/unit/test_smoke.py::test_version_string_present PASSED
     tests/unit/test_smoke.py::test_no_ml_imports_at_phase_0 PASSED
     tests/unit/test_smoke.py::test_package_in_src_layout PASSED

git commit
  -> [main (root-commit) 3276ccd] feat(phase-0): repository skeleton, environment, and guardrails
     21 files changed, 2865 insertions(+)
```

---

## Known issues / limitations

- Python 3.13 is the runtime; all future dependencies must be verified for 3.13 compatibility before adding them to `pyproject.toml`.
- `bitsandbytes` may not support Python 3.13 on Windows — must be verified at Phase 4/8 when quantisation is needed.
- `build-backend` is correctly set to `setuptools.build_meta` (initial draft had wrong value; corrected before first commit).

---

## Phase 0 acceptance criteria

| Criterion | Status |
|---|---|
| `python -c "import contextforge"` succeeds | VERIFIED |
| `pytest` succeeds | VERIFIED |
| No secrets committed | VERIFIED |
| Implementation plan present | VERIFIED |
| Project state identifies Phase 0 | VERIFIED |
| No LangChain/LlamaIndex/DB substitution | VERIFIED |
| `.gitignore` exists and protects secrets/artifacts | VERIFIED |
| `.env.example` exists without real credentials | VERIFIED |
| `pyproject.toml` (dependency file) exists | VERIFIED |
| `config/project.yaml` (project config) exists | VERIFIED |
| Planning/state documents exist | VERIFIED |
| Project name is ContextForge where appropriate | VERIFIED |
| No later-phase implementation introduced | VERIFIED |

---

## Next allowed task

> **Phase 1 — Domain and Dataset Contract**
>
> Gate: Human must choose a domain and approve corpus sources before any Phase 1 work begins.
> Record the choice in `docs/DECISIONS.md`.
