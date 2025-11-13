#!/usr/bin/env python3
"""Debug: Check what model is actually generating"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Config
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
ADAPTER = "slm_swap/04_ft/adapter_llama_structured"
TEST_FILE = "slm_swap/02_dataset/structured/test.jsonl"

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load ONE example
with open(TEST_FILE) as f:
    example = json.loads(f.readline())

print("\n" + "="*80)
print("PROMPT:")
print("="*80)
print(example["prompt"])

print("\n" + "="*80)
print("EXPECTED OUTPUT:")
print("="*80)
print(example["completion"])

# Generate with chat template
print("\n" + "="*80)
print("MODEL OUTPUT (with chat template):")
print("="*80)

messages = [{"role": "user", "content": example["prompt"]}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(prediction)

print("\n" + "="*80)
print("EXTRACTED COMPLETION:")
print("="*80)
completion_only = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(completion_only)
