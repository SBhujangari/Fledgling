"""Thin helpers to enforce Langfuse traces/events across the pipeline."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse

_REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
_langfuse_singleton: Optional[Langfuse] = None
_DISABLED = os.getenv("LANGFUSE_DISABLED", "").lower() in ("1", "true", "yes")


def _missing_env() -> list[str]:
    return [key for key in _REQUIRED_ENV if not os.getenv(key)]


def get_langfuse() -> Langfuse:
    """Return a memoized Langfuse client, enforcing required environment variables."""
    if _DISABLED:
        raise RuntimeError("Langfuse usage is disabled via LANGFUSE_DISABLED.")
    global _langfuse_singleton
    missing = _missing_env()
    if missing:
        raise EnvironmentError(
            f"Missing Langfuse environment variables: {', '.join(missing)}"
        )
    if _langfuse_singleton is None:
        _langfuse_singleton = Langfuse()
    return _langfuse_singleton


class _NoopTrace:
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {}

    def end(self, **_: Any):
        return None

    def update(self, **_: Any):
        return None


@contextmanager
def langfuse_trace(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager that records start/end for a Langfuse trace."""
    if _DISABLED:
        yield _NoopTrace(metadata=metadata)
        return

    client = get_langfuse()
    trace = client.trace(name=name, metadata=metadata or {})
    try:
        yield trace
        trace.end(output={"status": "completed"})
    except Exception as exc:  # pragma: no cover - pass-through for trace logging
        trace.end(
            output={"status": "error", "error": str(exc)},
        )
        raise


def log_decision_event(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
):
    """Log a Langfuse event for compare decisions."""
    if _DISABLED:
        return
    client = get_langfuse()
    client.event(name=name, metadata=metadata or {}, level=level)
