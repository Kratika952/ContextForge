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


def test_no_ml_imports_at_phase_0() -> None:
    """Phase 0 package must not silently pull in heavy ML libraries.

    This test documents intent: if torch, transformers, or similar are
    accidentally imported by the package init, this test catches it early.
    """
    import contextforge  # noqa: PLC0415,F401

    heavy_libs = ["torch", "transformers", "sentence_transformers", "faiss"]
    for lib in heavy_libs:
        assert lib not in sys.modules, (
            f"'{lib}' was imported by contextforge at Phase 0. "
            "Heavy ML imports must not appear in the package until the relevant phase."
        )


def test_package_in_src_layout() -> None:
    """Package must be loadable from the src/ layout."""
    import contextforge  # noqa: PLC0415

    pkg_file = str(contextforge.__file__).replace("\\", "/")
    assert "src/contextforge" in pkg_file, (
        f"Package file {pkg_file!r} does not appear to be under src/contextforge/. "
        "Check pyproject.toml [tool.setuptools.package-dir]."
    )
