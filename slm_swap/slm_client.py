"""Minimal local SLM client that performs greedy chat completions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from seed_utils import set_global_seed


def _ensure_loss_kwargs_stub() -> None:
    """Backfill transformers.utils.LossKwargs for older wheels."""

    try:
        from transformers.utils import LossKwargs  # type: ignore
    except ImportError:
        import transformers.utils as _tf_utils

        class LossKwargs(TypedDict, total=False):  # type: ignore
            """Empty TypedDict shim so Phi-4 remote modules import cleanly."""

            pass

        _tf_utils.LossKwargs = LossKwargs


_ensure_loss_kwargs_stub()

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency until training runs
    PeftModel = None  # type: ignore


@dataclass
class Message:
    role: str
    content: str


DEFAULT_SLM_MODEL_PATH = "models/phi-4-mini"


class SLMClient:
    """Loads a local SLM in 8-bit mode and serves greedy chat completions."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        # Set global seed for deterministic evaluation
        set_global_seed(seed)

        self.model_path = (
            model_path or os.getenv("SLM_MODEL_PATH") or DEFAULT_SLM_MODEL_PATH
        )
        if not self.model_path:
            raise EnvironmentError("SLM_MODEL_PATH must be set or passed explicitly.")
        if os.path.isdir(self.model_path):
            # Normalize relative paths so running from slm_swap/ works without surprises.
            self.model_path = os.path.abspath(self.model_path)
        else:
            raise FileNotFoundError(
                "Local SLM checkpoint not found at "
                f"{self.model_path}. If you're already inside slm_swap/, set "
                "SLM_MODEL_PATH=models/phi-4-mini or provide an absolute path."
            )
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if self.adapter_path:
            if PeftModel is None:
                raise ImportError("peft is required to load adapters.")
            if not os.path.isdir(self.adapter_path):
                raise FileNotFoundError(
                    f"Adapter directory not found: {self.adapter_path}"
                )
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        # Explicitly set model to evaluation mode (disables dropout, batch norm updates)
        self.model.eval()

    def generate(self, messages: List[dict]) -> str:
        """Execute greedy decoding for a single chat prompt."""
        return self.generate_batch([messages])[0]

    def generate_batch(self, batch_messages: List[List[dict]]) -> List[str]:
        """
        Execute greedy decoding for a batch of chat prompts in parallel.

        This leverages tensor parallelism across 4 GPUs (via device_map="auto")
        and batches examples together for maximum throughput.

        Args:
            batch_messages: List of message lists, where each message list is a conversation

        Returns:
            List of generated strings, one per conversation
        """
        # Convert all conversations to chat format
        chat_texts = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            for messages in batch_messages
        ]

        # Tokenize with padding to handle variable lengths
        inputs = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,  # Disable nucleus sampling
                top_k=0,  # Disable top-k sampling
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode each sequence, skipping the input portion
        results = []
        for i, output_ids in enumerate(outputs):
            input_length = inputs["input_ids"][i].shape[0]
            generated = output_ids[input_length:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(decoded)

        return results
