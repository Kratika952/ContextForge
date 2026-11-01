"""ContextForge UI package.
===========================
Phase 11: Streamlit frontend and decoupled HTTP client.
"""

from contextforge.ui.client import (
    ApiConnectionError,
    ApiError,
    ApiResponseError,
    ContextForgeClient,
)

__all__ = [
    "ApiConnectionError",
    "ApiError",
    "ApiResponseError",
    "ContextForgeClient",
]
