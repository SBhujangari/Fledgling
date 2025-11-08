"""Client factory for hosted Azure LLM and local SLM."""

from __future__ import annotations

from typing import Optional

from azure_client import AzureClient
from env_loader import ensure_env_loaded
from slm_client import SLMClient


def build_client(model_kind: str, adapter: Optional[str] = None):
    ensure_env_loaded()
    if model_kind == "slm":
        return SLMClient(adapter_path=adapter)
    if model_kind == "azure":
        return AzureClient()
    raise ValueError(f"Unsupported model kind: {model_kind}")
