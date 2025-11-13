#!/usr/bin/env python3
"""
Quick inference test script for API integration
Usage: python test_single_inference.py "Your prompt here" [model_type]
"""

import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
ADAPTERS = {
    "structured": "slm_swap/04_ft/adapter_llama_structured",
    "hermes": "slm_swap/04_ft/adapter_llama_hermes",
    "cep": "slm_swap/04_ft/adapter_llama_cep"
}

def load_model(adapter_type="structured"):
    """Load the fine-tuned model"""
    if adapter_type not in ADAPTERS:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    adapter_path = ADAPTERS[adapter_type]

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate(model, tokenizer, prompt: str) -> str:
    """Generate completion for prompt"""
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
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    completion = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return completion.strip()

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: python test_single_inference.py \"prompt\" [model_type]"
        }))
        sys.exit(1)

    prompt = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "structured"

    try:
        print(f"Loading {model_type} model...", file=sys.stderr)
        model, tokenizer = load_model(model_type)

        print("Generating...", file=sys.stderr)
        result = generate(model, tokenizer, prompt)

        # Output result as JSON
        output = {
            "prompt": prompt,
            "model": model_type,
            "result": result,
            "success": True
        }

        # Try to parse as JSON if it looks like JSON
        if result.strip().startswith('{'):
            try:
                parsed = json.loads(result)
                output["parsed_result"] = parsed
            except json.JSONDecodeError:
                pass

        print(json.dumps(output, indent=2))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "success": False
        }), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
