"""Utility to load repository-level environment variables exactly once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv

_ENV_LOADED = False

DEFAULT_AZURE_VERSION = "2024-05-01-preview"
DEFAULT_AZURE_DEPLOYMENT = "gpt-oss-120b"
DEFAULT_LANGFUSE_DISABLED = "1"


def _resolve_dotenv() -> Optional[str]:
    """Locate the closest .env (repo root is one level up from slm_swap/)."""
    # find_dotenv walks up the tree starting from cwd; explicitly seed with repo root
    repo_root = Path(__file__).resolve().parent.parent
    env_path = find_dotenv(filename=".env", usecwd=True)
    if env_path:
        return env_path
    candidate = repo_root / ".env"
    return str(candidate) if candidate.exists() else None


def ensure_env_loaded() -> None:
    """Load .env once and normalize key names/defaults for downstream clients."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = _resolve_dotenv()
    if env_path:
        load_dotenv(env_path, override=False)

    # Normalize Azure naming so AZURE_OPENAI_KEY works without duplication.
    # Remove ALL whitespace from the key to prevent "Illegal header value" errors.
    # This handles cases where the key might have embedded newlines or spaces.
    if not os.getenv("AZURE_API_KEY"):
        alt = os.getenv("AZURE_OPENAI_KEY")
        if alt:
            os.environ["AZURE_API_KEY"] = "".join(alt.split())
    else:
        # Also remove whitespace if AZURE_API_KEY is already set
        existing = os.getenv("AZURE_API_KEY")
        if existing:
            os.environ["AZURE_API_KEY"] = "".join(existing.split())

    # Strip leading/trailing whitespace from AZURE_ENDPOINT to prevent parsing issues
    endpoint = os.getenv("AZURE_ENDPOINT")
    if endpoint:
        os.environ["AZURE_ENDPOINT"] = endpoint.strip()

    # Provide safe defaults so local runs work with the repo sample config.
    os.environ.setdefault("AZURE_API_VERSION", DEFAULT_AZURE_VERSION)
    os.environ.setdefault("AZURE_DEPLOYMENT", DEFAULT_AZURE_DEPLOYMENT)
    os.environ.setdefault("LANGFUSE_DISABLED", DEFAULT_LANGFUSE_DISABLED)

    _ENV_LOADED = True
