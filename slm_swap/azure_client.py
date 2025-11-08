"""Azure OpenAI-compatible client for hosted instructor baseline."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse

import requests
from openai import AzureOpenAI


def _parse_endpoint(endpoint: str) -> Tuple[str, Optional[str], bool]:
    """Split endpoint into base URL and deployment name."""
    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    deployment = None
    parts = [p for p in parsed.path.split("/") if p]
    use_models_api = "models" in parts
    if "deployments" in parts:
        idx = parts.index("deployments")
        if idx + 1 < len(parts):
            deployment = parts[idx + 1]
    return base, deployment, use_models_api


class AzureClient:
    """Minimal wrapper around Azure's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment: Optional[str] = None,
        temperature: float = 0.0,
    ):
        endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_API_KEY")
        api_version = api_version or os.getenv("AZURE_API_VERSION")
        env_deployment = os.getenv("AZURE_DEPLOYMENT")

        if not endpoint or not api_key or not api_version:
            raise EnvironmentError(
                "AZURE_ENDPOINT, AZURE_API_KEY, and AZURE_API_VERSION must be set."
            )

        base_endpoint, parsed_deployment, use_models_api = _parse_endpoint(endpoint)
        self.temperature = temperature
        self.use_models_api = use_models_api
        self.api_key = api_key

        if use_models_api:
            self.client = None
            parsed = urlparse(endpoint)
            query = parsed.query or urlencode({"api-version": api_version})
            self.models_url = urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment)
            )
        else:
            deployment = deployment or env_deployment or parsed_deployment
            if not deployment:
                raise EnvironmentError(
                    "Deployment name missing. Pass AZURE_DEPLOYMENT or include it in AZURE_ENDPOINT."
                )
            self.deployment = deployment
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_endpoint,
            )

    def generate(self, messages: List[dict]) -> str:
        """Invoke Azure chat completion with temperature=0."""
        if self.use_models_api:
            resp = requests.post(
                self.models_url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                },
                json={
                    "messages": messages,
                    "temperature": self.temperature,
                },
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            choice = payload["choices"][0]["message"]
            return (choice.get("content") or "").strip()

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=self.temperature,
            messages=messages,
        )
        choice = response.choices[0].message
        return (choice.content or "").strip()
