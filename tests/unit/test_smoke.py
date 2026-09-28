# Phase 0 smoke test — verifies that the contextforge package imports correctly
# and exposes the expected package metadata.
#
# This is the minimal test required before Phase 1 begins.
# No ML dependencies are used here.

import importlib
import sys


def test_package_importable() -> None:
    """contextforge must import without errors."""
    mod = importlib.import_module("contextforge")
    assert mod is not None


def test_version_string_present() -> None:
    """contextforge.__version__ must be a non-empty string."""
    import contextforge  # noqa: PLC0415

    assert isinstance(contextforge.__version__, str)
    assert len(contextforge.__version__) > 0


import subprocess


def test_no_ml_imports_at_phase_0() -> None:
    """Phase 0 package must not silently pull in heavy ML libraries.

    This test documents intent: if torch, transformers, or similar are
    accidentally imported by the package init, this test catches it early.
    Runs in an isolated process to ensure no pollution from retrieval tests.
    """
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys, contextforge; "
            "heavy = ['torch', 'transformers', 'sentence_transformers', 'faiss']; "
            "loaded = [m for m in heavy if m in sys.modules]; "
            "assert not loaded, f'Heavy ML libraries imported by contextforge: {loaded}'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


def test_package_in_src_layout() -> None:
    """Package must be loadable from the src/ layout."""
    import contextforge  # noqa: PLC0415

    pkg_file = str(contextforge.__file__).replace("\\", "/")
    assert "src/contextforge" in pkg_file, (
        f"Package file {pkg_file!r} does not appear to be under src/contextforge/. "
        "Check pyproject.toml [tool.setuptools.package-dir]."
    )
