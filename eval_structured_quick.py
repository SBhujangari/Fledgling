#!/usr/bin/env python3
"""Quick evaluation for structured format with correct base model"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Config
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
ADAPTER = "slm_swap/04_ft/adapter_llama_structured"
TEST_FILE = "slm_swap/02_dataset/structured/test.jsonl"

# Load model
print(f"Loading {BASE_MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

print(f"Loading adapter from {ADAPTER}...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load test data
print(f"Loading test data...")
test_examples = []
with open(TEST_FILE) as f:
    for line in f:
        test_examples.append(json.loads(line))

print(f"Evaluating {len(test_examples)} examples...")

# Evaluate
correct = 0
total = 0

for ex in tqdm(test_examples):
    prompt = ex["prompt"]
    expected = ex["completion"]

    # Generate with chat template
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.0,  # Greedy
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    prediction = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

    # Compare
    try:
        pred_json = json.loads(prediction)
        exp_json = json.loads(expected)

        # Check if they match
        if pred_json == exp_json:
            correct += 1
    except:
        pass  # Invalid JSON

    total += 1

# Results
accuracy = correct / total if total > 0 else 0

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Total examples: {total}")
print(f"Exact matches: {correct}")
print(f"Accuracy: {accuracy:.1%}")
print(f"{'='*60}")

# Save results
results = {
    "total": total,
    "correct": correct,
    "accuracy": accuracy
}

with open("eval_structured_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: eval_structured_results.json")
