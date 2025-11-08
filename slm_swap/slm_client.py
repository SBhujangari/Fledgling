"""Minimal local SLM client that performs greedy chat completions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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
    """Loads a local SLM in 4-bit mode and serves greedy chat completions."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        self.model_path = (
            model_path or os.getenv("SLM_MODEL_PATH") or DEFAULT_SLM_MODEL_PATH
        )
        if not self.model_path:
            raise EnvironmentError("SLM_MODEL_PATH must be set or passed explicitly.")
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
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

    def generate(self, messages: List[dict]) -> str:
        """Execute greedy decoding for a chat prompt."""
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
