#!/usr/bin/env python3
"""
ContextForge Ingestion CLI.
===========================
Usage:
    python scripts/ingest.py <document-or-directory-path> [--db <path>] [--id <custom-id>] [--title <custom-title>]

Examples:
    python scripts/ingest.py data/raw/nist-csf-2.0.pdf
    python scripts/ingest.py data/raw/owasp-top10-2025/
    python scripts/ingest.py data/raw/cisa-cpg-2.0.pdf --db data/test.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if not installed
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from contextforge.common.config import get_db_path
from contextforge.common.types import ContextForgeError
from contextforge.ingestion.service import IngestionResult, IngestionService


def print_result(result: IngestionResult, db_path: Path | str) -> None:
    """Print structured summary of an ingestion result."""
    doc = result.document
    meta = doc.metadata or {}
    total_units = meta.get("total_units", "N/A")
    total_chars = meta.get("total_chars", "N/A")

    print("-" * 50)
    print(f"Status:         {result.status}")
    print(f"Document ID:    {doc.id}")
    print(f"Title:          {doc.title}")
    print(f"Source type:    {doc.source_type.upper()}")
    print(f"SHA-256:        {doc.checksum}")
    print(f"Pages/sections: {total_units}")
    print(f"Characters:     {total_chars}")
    print(f"Database:       {db_path}")
    if result.message:
        print(f"Notes:          {result.message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents (.pdf, .md) into ContextForge SQLite metadata store."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a document file (.pdf, .md) or directory of documents.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional path to target SQLite database (defaults to project config).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Optional custom document ID (single file only).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom document title (single file only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, raises error instead of skipping duplicate documents.",
    )

    args = parser.parse_args()
    target_path = Path(args.path).resolve()
    db_target = Path(args.db).resolve() if args.db else get_db_path()

    if not target_path.exists():
        print(f"Status: FAILED")
        print(f"Reason: Path not found: {target_path}")
        return 1

    on_duplicate = "raise" if args.force else "skip"

    try:
        with IngestionService(db_path=db_target) as service:
            if target_path.is_file():
                result = service.ingest_file(
                    target_path,
                    document_id=args.id,
                    title=args.title,
                    on_duplicate=on_duplicate,
                )
                print_result(result, db_target)
                print("-" * 50)
                return 0
            elif target_path.is_dir():
                results = service.ingest_directory(
                    target_path,
                    recursive=True,
                    on_duplicate=on_duplicate,
                )
                if not results:
                    print(f"Status: FAILED")
                    print(f"Reason: No supported document files (.pdf, .md) found in {target_path}")
                    return 1

                new_count = sum(1 for r in results if r.is_new)
                dup_count = sum(1 for r in results if not r.is_new)
                for r in results:
                    print_result(r, db_target)
                print("-" * 50)
                print(f"Ingestion batch complete: {new_count} newly ingested, {dup_count} duplicate(s) skipped.")
                return 0
            else:
                print(f"Status: FAILED")
                print(f"Reason: Target path is neither a regular file nor a directory: {target_path}")
                return 1

    except ContextForgeError as exc:
        print(f"Status: FAILED")
        print(f"Reason: [{type(exc).__name__}] {exc}")
        return 1
    except Exception as exc:
        print(f"Status: FAILED")
        print(f"Reason: Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
