#!/usr/bin/env python3
"""
Evaluation script for function calling (Paper approach)
Evaluates against Hermes Function Calling dataset
"""

import os
import json
import argparse
import re
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm
from jsonschema import validate, ValidationError


# Set deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def extract_json_from_text(text):
    """Extract JSON object from model output"""
    # Try to find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def evaluate_function_call(pred_tool_call, true_tool_call):
    """
    Evaluate a single function call prediction
    Returns dict with metrics
    """
    metrics = {
        "valid_json": False,
        "name_match": False,
        "args_exact_match": False,
        "args_field_precision": 0.0,
        "args_field_recall": 0.0,
        "args_field_f1": 0.0,
    }

    # Check if prediction is valid JSON
    if pred_tool_call is None:
        return metrics

    metrics["valid_json"] = True

    # Check tool name match
    pred_name = pred_tool_call.get("name", "")
    true_name = true_tool_call.get("name", "")
    metrics["name_match"] = (pred_name == true_name)

    # Check arguments
    pred_args = pred_tool_call.get("arguments", {})
    true_args = true_tool_call.get("arguments", {})

    # Exact match (canonical JSON)
    pred_canonical = json.dumps(pred_args, sort_keys=True, ensure_ascii=False)
    true_canonical = json.dumps(true_args, sort_keys=True, ensure_ascii=False)
    metrics["args_exact_match"] = (pred_canonical == true_canonical)

    # Field-level metrics
    if isinstance(pred_args, dict) and isinstance(true_args, dict):
        pred_keys = set(pred_args.keys())
        true_keys = set(true_args.keys())

        if len(pred_keys) > 0:
            # Count matching key-value pairs
            matching = 0
            for key in pred_keys:
                if key in true_args and pred_args[key] == true_args[key]:
                    matching += 1

            metrics["args_field_precision"] = matching / len(pred_keys) if len(pred_keys) > 0 else 0.0

        if len(true_keys) > 0:
            matching = 0
            for key in true_keys:
                if key in pred_args and pred_args[key] == true_args[key]:
                    matching += 1

            metrics["args_field_recall"] = matching / len(true_keys) if len(true_keys) > 0 else 0.0

        # F1
        prec = metrics["args_field_precision"]
        rec = metrics["args_field_recall"]
        if prec + rec > 0:
            metrics["args_field_f1"] = 2 * prec * rec / (prec + rec)

    return metrics


def format_prompt(example):
    """Format Hermes example into prompt (actual format)"""
    conversations = example.get("conversations", [])

    # Build prompt from system and human messages only
    prompt = ""
    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            prompt += f"<|system|>\n{value}\n\n"
        elif from_role == "human":
            prompt += f"<|user|>\n{value}\n\n"
        # Skip gpt response - we want model to generate it

    prompt += "<|assistant|>\n"
    return prompt


def conversations_to_messages(example):
    """Convert Hermes conversations format to chat template messages format"""
    conversations = example.get("conversations", [])
    messages = []

    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            messages.append({"role": "system", "content": value})
        elif from_role == "human":
            messages.append({"role": "user", "content": value})
        # Skip gpt response - we want model to generate it

    return messages


def main():
    parser = argparse.ArgumentParser(description="Evaluate function calling model")
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "microsoft/phi-4-mini"),
                        help="Base model ID")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--dataset-path", type=str, default="datasets/hermes_function_calling_test.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--output-path", type=str, default="eval_results/results.json",
                        help="Output path for results")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max new tokens (default: 256)")

    args = parser.parse_args()

    print("=" * 60)
    print("FUNCTION_CALLING_FINETUNE - Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load adapter if provided
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Test examples: {len(dataset)}")

    # Evaluate
    print("\nEvaluating...")
    all_metrics = []
    details = []

    for example in tqdm(dataset, desc="Evaluating"):
        # Format prompt using the SAME format as training (custom tokens, not chat template)
        prompt = format_prompt(example)

        # Generate (use raw text tokenization, NOT chat template)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and extract response after <|assistant|> marker
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text.split("<|assistant|>")[-1].strip()

        # Extract JSON from prediction
        pred_tool_call = extract_json_from_text(response)

        # Extract true answer from conversations (gpt's response)
        conversations = example.get("conversations", [])
        true_response = ""
        for msg in conversations:
            if msg.get("from") == "gpt":
                true_response = msg.get("value", "")
                break
        true_tool_call = extract_json_from_text(true_response)

        # Evaluate
        metrics = evaluate_function_call(pred_tool_call, true_tool_call)
        all_metrics.append(metrics)

        # Store details
        details.append({
            "prompt": prompt,
            "prediction": response,
            "pred_tool_call": pred_tool_call,
            "true_tool_call": true_tool_call,
            "metrics": metrics,
        })

    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = {
        "valid_json_rate": sum(m["valid_json"] for m in all_metrics) / len(all_metrics),
        "name_match_rate": sum(m["name_match"] for m in all_metrics) / len(all_metrics),
        "args_exact_match_rate": sum(m["args_exact_match"] for m in all_metrics) / len(all_metrics),
        "args_field_precision": sum(m["args_field_precision"] for m in all_metrics) / len(all_metrics),
        "args_field_recall": sum(m["args_field_recall"] for m in all_metrics) / len(all_metrics),
        "args_field_f1": sum(m["args_field_f1"] for m in all_metrics) / len(all_metrics),
    }

    # Create result object
    result = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "dataset_path": args.dataset_path,
        "num_examples": len(dataset),
        "timestamp": datetime.now().isoformat(),
        "metrics": aggregated,
        "approach": "FUNCTION_CALLING_FINETUNE_paper",
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save details
    details_path = args.output_path.replace(".json", "_details.jsonl")
    with open(details_path, "w") as f:
        for detail in details:
            f.write(json.dumps(detail) + "\n")

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in aggregated.items():
        print(f"{key}: {value:.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_path}")
    print(f"Details saved to: {details_path}")


if __name__ == "__main__":
    main()
