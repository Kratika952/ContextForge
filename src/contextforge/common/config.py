"""
Configuration loader for ContextForge.
=======================================
Reads project settings from config/project.yaml with environment overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the absolute path to the ContextForge project root."""
    # This file is located at <project_root>/src/contextforge/common/config.py
    return Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """Return the data directory path."""
    custom = os.environ.get("CONTEXTFORGE_DATA_DIR")
    if custom:
        return Path(custom)
    return get_project_root() / "data"


def get_db_path() -> Path:
    """Return the SQLite database path."""
    custom = os.environ.get("CONTEXTFORGE_DB_PATH")
    if custom:
        return Path(custom)
    return get_data_dir() / "contextforge.db"


def get_raw_data_dir() -> Path:
    """Return the raw data directory path."""
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """Return the processed data directory path."""
    return get_data_dir() / "processed"


def get_manifests_dir() -> Path:
    """Return the manifests directory path."""
    return get_data_dir() / "manifests"
