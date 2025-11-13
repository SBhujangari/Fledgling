"""OpenAI API client for GPT models."""

from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

from env_loader import ensure_env_loaded


class OpenAIClient:
    """
    OpenAI API client wrapper.

    Uses GPT-5 (or other specified model) for high-quality responses.
    Primarily used for demo purposes and comparison baselines.
    """

    def __init__(
        self,
        model: str = "gpt-4o",  # Default to gpt-4o since gpt-5 may not be available
        api_key: Optional[str] = None,
        temperature: float = 1.0,  # GPT-5 only supports default temperature of 1.0
        max_tokens: int = 2000,
    ):
        ensure_env_loaded()

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature (if supported by model)
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        # Use instance defaults if not specified
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Prepare API call parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature if not using default
        if temp != 1.0:
            params["temperature"] = temp

        # Use max_completion_tokens for newer models, max_tokens for older
        if self.model.startswith("gpt-5") or self.model.startswith("o1"):
            params["max_completion_tokens"] = tokens
        else:
            params["max_tokens"] = tokens

        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model})"
