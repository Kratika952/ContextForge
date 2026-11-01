"""HTTP Client for ContextForge FastAPI Backend.
=============================================
Phase 11: Decoupled client communicating with the backend via HTTP.
Does NOT import causal models, pipelines, FAISS, or CUDA.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(Exception):
    """Base exception for API communication failures."""


class ApiConnectionError(ApiError):
    """Raised when the backend API is unreachable."""


class ApiResponseError(ApiError):
    """Raised when the API returns an HTTP error status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API Error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ContextForgeClient:
    """Lightweight HTTP client communicating with the ContextForge FastAPI backend."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0) -> None:
        """Initialize client with configured base URL.

        Args:
            base_url: Optional API URL override. If omitted, uses CONTEXTFORGE_API_URL or default.
            timeout: Network request timeout in seconds.
        """
        env_url = os.getenv("CONTEXTFORGE_API_URL", "").strip()
        self.base_url = (base_url or env_url or DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> dict[str, Any]:
        """Query the /health endpoint.

        Returns:
            Parsed health response dictionary.

        Raises:
            ApiConnectionError: If the server cannot be reached.
            ApiResponseError: If the server returns a non-200 code.
        """
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "ContextForge-UI/0.1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ApiResponseError(e.code, body) from e
        except urllib.error.URLError as e:
            raise ApiConnectionError(
                f"Could not connect to ContextForge API at {self.base_url}. "
                "Please verify that the FastAPI backend server is running."
            ) from e
        except Exception as e:
            raise ApiError(f"Unexpected error contacting health endpoint: {e}") from e

    def ask(self, question: str, pipeline: str) -> dict[str, Any]:
        """Submit a question to the /ask endpoint.

        Args:
            question: User question string.
            pipeline: Target pipeline ('base', 'rag', or 'finetuned').

        Returns:
            Parsed AskResponse dictionary matching AnswerResult contract.

        Raises:
            ValueError: If inputs are invalid before sending.
            ApiConnectionError: If the backend cannot be reached.
            ApiResponseError: If the backend returns an error response.
        """
        cleaned_q = question.strip()
        if not cleaned_q:
            raise ValueError("Question cannot be empty or whitespace-only.")

        valid_pipelines = ("base", "rag", "finetuned")
        clean_pipe = pipeline.lower().strip()
        if clean_pipe not in valid_pipelines:
            raise ValueError(f"Invalid pipeline '{pipeline}'. Supported: {valid_pipelines}")

        url = f"{self.base_url}/ask"
        payload = json.dumps({"question": cleaned_q, "pipeline": clean_pipe}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ContextForge-UI/0.1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            detail = body
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and "detail" in parsed:
                    detail = str(parsed["detail"])
            except Exception:
                pass
            raise ApiResponseError(e.code, detail) from e
        except urllib.error.URLError as e:
            raise ApiConnectionError(
                f"Connection failed to {self.base_url}/ask: {e.reason}. "
                "Ensure FastAPI backend is active and reachable."
            ) from e
        except Exception as e:
            raise ApiError(f"Unexpected error sending question: {e}") from e
