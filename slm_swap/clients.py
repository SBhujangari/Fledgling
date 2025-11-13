"""Client factory for hosted Azure LLM, OpenAI GPT models, and local SLM."""

from __future__ import annotations

from typing import Optional

from azure_client import AzureClient
from env_loader import ensure_env_loaded
from openai_client import OpenAIClient
from slm_client import SLMClient


def build_client(model_kind: str, adapter: Optional[str] = None):
    ensure_env_loaded()
    if model_kind == "slm":
        return SLMClient(adapter_path=adapter)
    if model_kind == "azure":
        return AzureClient()
    if model_kind == "openai":
        return OpenAIClient(model="gpt-4o")  # Use gpt-4o as default
    raise ValueError(f"Unsupported model kind: {model_kind}")
