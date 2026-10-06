"""Phase 7 — Fine-tuning dataset unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# ── Import the builder module directly ───────────────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_finetuning_dataset import (
    DATASET,
    VALID_SOURCES,
    SYSTEM_PROMPT,
    _build_records,
    _check_duplicates,
    _check_fake_citations,
    _check_leakage,
    _compute_stats,
    _deterministic_split,
    _load_evaluation_questions,
    _normalize_question,
    _select_review_sample,
)

QA_DIR = PROJECT_ROOT / "data" / "qa"


# ── Record structure ─────────────────────────────────────────────────────────


class TestRecordStructure:
    """Every record must have the correct schema."""

    @pytest.fixture(scope="class")
    def records(self) -> list[dict]:
        return _build_records()

    def test_total_count_matches_dataset(self, records: list[dict]) -> None:
        assert len(records) == len(DATASET)

    def test_record_has_required_keys(self, records: list[dict]) -> None:
        required = {"id", "messages", "source_family", "source_document_id", "category"}
        for rec in records:
            assert required.issubset(rec.keys()), f"{rec['id']} missing keys"

    def test_messages_are_system_user_assistant(self, records: list[dict]) -> None:
        for rec in records:
            roles = [m["role"] for m in rec["messages"]]
            assert roles == ["system", "user", "assistant"], f"{rec['id']} wrong roles"

    def test_system_prompt_is_consistent(self, records: list[dict]) -> None:
        for rec in records:
            assert rec["messages"][0]["content"] == SYSTEM_PROMPT

    def test_ids_are_unique(self, records: list[dict]) -> None:
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_no_empty_questions_or_answers(self, records: list[dict]) -> None:
        for rec in records:
            assert len(rec["messages"][1]["content"].strip()) > 0
            assert len(rec["messages"][2]["content"].strip()) > 0


# ── Source provenance ────────────────────────────────────────────────────────


class TestSourceProvenance:
    """Every record references a valid source document."""

    @pytest.fixture(scope="class")
    def records(self) -> list[dict]:
        return _build_records()

    def test_all_source_families_valid(self, records: list[dict]) -> None:
        valid_families = set(VALID_SOURCES.keys())
        for rec in records:
            assert rec["source_family"] in valid_families, f"{rec['id']}"

    def test_all_source_doc_ids_valid(self, records: list[dict]) -> None:
        all_ids = set()
        for ids in VALID_SOURCES.values():
            all_ids.update(ids)
        for rec in records:
            assert rec["source_document_id"] in all_ids, f"{rec['id']}"


# ── Leakage and duplicates ──────────────────────────────────────────────────


class TestLeakageAndDuplicates:
    """No overlap with evaluation set; no internal duplicates."""

    @pytest.fixture(scope="class")
    def records(self) -> list[dict]:
        return _build_records()

    def test_no_evaluation_leakage(self, records: list[dict]) -> None:
        eval_qs = _load_evaluation_questions()
        leaks = _check_leakage(records, eval_qs)
        assert leaks == [], f"Evaluation leakage: {leaks}"

    def test_no_duplicate_questions(self, records: list[dict]) -> None:
        dupes = _check_duplicates(records)
        assert dupes == [], f"Duplicates: {dupes}"

    def test_no_fake_citations(self, records: list[dict]) -> None:
        fakes = _check_fake_citations(records)
        assert fakes == [], f"Fake citations: {fakes}"


# ── Splitting ────────────────────────────────────────────────────────────────


class TestSplitting:
    """Deterministic split with no overlap."""

    @pytest.fixture(scope="class")
    def splits(self) -> tuple[list[dict], list[dict], list[dict]]:
        return _deterministic_split(_build_records())

    def test_split_sizes_sum(self, splits) -> None:
        train, val, test = splits
        assert len(train) + len(val) + len(test) == len(DATASET)

    def test_no_split_overlap(self, splits) -> None:
        train, val, test = splits
        train_ids = {r["id"] for r in train}
        val_ids = {r["id"] for r in val}
        test_ids = {r["id"] for r in test}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_split_is_deterministic(self) -> None:
        records = _build_records()
        t1, v1, s1 = _deterministic_split(records)
        t2, v2, s2 = _deterministic_split(records)
        assert [r["id"] for r in t1] == [r["id"] for r in t2]
        assert [r["id"] for r in v1] == [r["id"] for r in v2]
        assert [r["id"] for r in s1] == [r["id"] for r in s2]

    def test_train_is_largest_split(self, splits) -> None:
        train, val, test = splits
        assert len(train) >= len(val)
        assert len(train) >= len(test)


# ── File outputs ─────────────────────────────────────────────────────────────


class TestFileOutputs:
    """Output files produced by the builder exist and are valid JSONL."""

    EXPECTED_FILES = [
        "finetune_full.jsonl",
        "finetune_train.jsonl",
        "finetune_val.jsonl",
        "finetune_test.jsonl",
        "finetune_review_sample.jsonl",
        "finetune_stats.json",
        "finetune_metadata.json",
    ]

    def test_all_output_files_exist(self) -> None:
        for name in self.EXPECTED_FILES:
            path = QA_DIR / name
            assert path.exists(), f"Missing: {name}"

    def test_jsonl_files_parseable(self) -> None:
        for name in self.EXPECTED_FILES:
            if not name.endswith(".jsonl"):
                continue
            path = QA_DIR / name
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"{name} line {i} is not valid JSON")

    def test_stats_file_has_total(self) -> None:
        with open(QA_DIR / "finetune_stats.json", "r") as f:
            stats = json.load(f)
        assert stats["total"] == len(DATASET)

    def test_metadata_file_has_version(self) -> None:
        with open(QA_DIR / "finetune_metadata.json", "r") as f:
            meta = json.load(f)
        assert "version" in meta
        assert meta["phase"] == 7

    def test_review_sample_balanced(self) -> None:
        with open(QA_DIR / "finetune_review_sample.jsonl", "r") as f:
            sample = [json.loads(line) for line in f if line.strip()]
        families = {r["source_family"] for r in sample}
        assert "owasp" in families
        assert "nist" in families
        assert "cisa" in families


# ── Stats helper ─────────────────────────────────────────────────────────────


class TestStatsComputation:
    """Stats function returns correct shape."""

    def test_stats_shape(self) -> None:
        records = _build_records()
        train, val, test = _deterministic_split(records)
        stats = _compute_stats(records, train, val, test)
        assert "total" in stats
        assert "splits" in stats
        assert set(stats["splits"].keys()) == {"train", "validation", "test"}
        for split in stats["splits"].values():
            assert "count" in split
