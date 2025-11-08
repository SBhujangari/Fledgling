"""Azure OpenAI-compatible client for hosted instructor baseline."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse

import warnings

import requests
from openai import AzureOpenAI


VALID_AZURE_HOST_FRAGMENT = ".openai.azure."
SERVICES_AZURE_HOST_SUFFIX = ".services.ai.azure.com"
OPENAI_HOST_SUFFIX = ".openai.azure.com"


def _normalize_netloc(parsed, warn: bool = True) -> str:
    """Return normalized host[:port] and ensure it targets Azure OpenAI."""
    host = parsed.hostname or ""
    if host and any(ch.isspace() for ch in host):
        cleaned = "".join(host.split())
        if not cleaned:
            raise EnvironmentError("AZURE_ENDPOINT host cannot be blank.")
        if warn:
            warnings.warn(
                f"Removed whitespace from Azure host '{parsed.netloc}'. "
                "Update AZURE_ENDPOINT to the canonical https://<resource>.openai.azure.com form.",
                RuntimeWarning,
                stacklevel=2,
            )
        host = cleaned
    lower = host.lower()
    if lower.endswith(SERVICES_AZURE_HOST_SUFFIX):
        resource = host[: -len(SERVICES_AZURE_HOST_SUFFIX)]
        host = f"{resource}{OPENAI_HOST_SUFFIX}"
        lower = host.lower()
        if warn:
            warnings.warn(
                f"Translated Azure AI host '{parsed.netloc}' to '{host}'. "
                "Update AZURE_ENDPOINT to the canonical https://<resource>.openai.azure.com form.",
                RuntimeWarning,
                stacklevel=2,
            )
    if VALID_AZURE_HOST_FRAGMENT not in lower:
        raise EnvironmentError(
            "AZURE_ENDPOINT must point at Azure OpenAI, e.g. https://<resource>.openai.azure.com. "
            f"Got host '{parsed.netloc}'."
        )
    return host if not parsed.port else f"{host}:{parsed.port}"


def _parse_endpoint(endpoint: str) -> Tuple[str, Optional[str], bool]:
    """Split endpoint into base URL and deployment name."""
    parsed = urlparse(endpoint)
    netloc = _normalize_netloc(parsed)
    base = f"{parsed.scheme}://{netloc}"
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
        seed: Optional[int] = 42,
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
        self.seed = seed
        self.use_models_api = use_models_api
        self.api_key = api_key

        if use_models_api:
            self.client = None
            parsed = urlparse(endpoint)
            netloc = _normalize_netloc(parsed, warn=False)
            query = parsed.query or urlencode({"api-version": api_version})
            self.models_url = urlunparse(
                (parsed.scheme, netloc, parsed.path, parsed.params, query, parsed.fragment)
            )
            self.models_model = deployment or env_deployment
            if not self.models_model:
                raise EnvironmentError(
                    "Provide AZURE_DEPLOYMENT (model slug) when using the models/chat/completions endpoint."
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
        """Invoke Azure chat completion with temperature=0 and fixed seed."""
        if self.use_models_api:
            payload = {
                "messages": messages,
                "temperature": self.temperature,
                "model": self.models_model,
            }
            if self.seed is not None:
                payload["seed"] = self.seed
            resp = requests.post(
                self.models_url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                },
                json=payload,
                timeout=60,
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - adds server details
                detail = ""
                if exc.response is not None:
                    detail = exc.response.text
                    try:
                        payload = exc.response.json()
                    except ValueError:
                        payload = None
                    else:
                        error = payload.get("error") if isinstance(payload, dict) else None
                        code = (error or {}).get("code")
                        if code == "content_filter":
                            # Azure filtered the prompt; surface a sentinel string so metrics mark invalid.
                            return "CONTENT_FILTERED"
                raise requests.HTTPError(
                    f"{exc} | response body: {detail}"
                ) from exc
            payload = resp.json()
            choice = payload["choices"][0]["message"]
            return (choice.get("content") or "").strip()

        kwargs = {
            "model": self.deployment,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        return (choice.content or "").strip()
