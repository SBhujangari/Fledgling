#!/usr/bin/env python3
"""
Detailed evaluation with comprehensive metrics to show model quality.
This script provides granular breakdowns beyond simple exact match accuracy.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Any
from collections import defaultdict

# Config
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"

# Use HuggingFace model by default, fall back to local if USE_LOCAL_MODEL is set
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
ADAPTER = "slm_swap/04_ft/adapter_llama_structured" if USE_LOCAL_MODEL else "kineticdrive/llama-structured-api-adapter"

TEST_FILE = "slm_swap/02_dataset/structured/test.jsonl"

print(f"📦 Loading adapter from: {'Local' if USE_LOCAL_MODEL else 'HuggingFace'} ({ADAPTER})")

def load_model():
    """Load fine-tuned model"""
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

    return model, tokenizer

def generate_prediction(model, tokenizer, prompt: str) -> str:
    """Generate prediction using chat template"""
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

def evaluate_example(expected: Dict, predicted: Dict) -> Dict[str, Any]:
    """Detailed evaluation of a single example"""
    metrics = {
        "exact_match": expected == predicted,
        "json_valid": isinstance(predicted, dict),
        "tool_name_match": False,
        "query_match": False,
        "args_exact_match": False,
        "args_keys_match": False,
        "args_partial_match": 0.0,
        "has_required_fields": False
    }

    if not metrics["json_valid"]:
        return metrics

    # Check required fields
    required_fields = {"query", "tool_name", "arguments"}
    metrics["has_required_fields"] = required_fields.issubset(predicted.keys())

    # Tool name match
    if "tool_name" in predicted and "tool_name" in expected:
        metrics["tool_name_match"] = predicted["tool_name"] == expected["tool_name"]

    # Query match
    if "query" in predicted and "query" in expected:
        metrics["query_match"] = predicted["query"] == expected["query"]
        # Also check case-insensitive and whitespace-normalized
        if not metrics["query_match"]:
            pred_q = predicted["query"].lower().strip()
            exp_q = expected["query"].lower().strip()
            metrics["query_normalized_match"] = pred_q == exp_q

    # Arguments evaluation
    if "arguments" in predicted and "arguments" in expected:
        pred_args = predicted["arguments"]
        exp_args = expected["arguments"]

        if isinstance(pred_args, dict) and isinstance(exp_args, dict):
            metrics["args_exact_match"] = pred_args == exp_args

            # Key match
            pred_keys = set(pred_args.keys())
            exp_keys = set(exp_args.keys())
            metrics["args_keys_match"] = pred_keys == exp_keys
            metrics["args_missing_keys"] = list(exp_keys - pred_keys)
            metrics["args_extra_keys"] = list(pred_keys - exp_keys)

            # Partial match (ratio of matching key-value pairs)
            if exp_keys:
                matching_pairs = 0
                for key in exp_keys:
                    if key in pred_args:
                        # Normalize values for comparison
                        pred_val = str(pred_args[key]).lower()
                        exp_val = str(exp_args[key]).lower()
                        if pred_val == exp_val:
                            matching_pairs += 1
                metrics["args_partial_match"] = matching_pairs / len(exp_keys)

    return metrics

def aggregate_metrics(all_metrics: List[Dict]) -> Dict[str, Any]:
    """Aggregate metrics across all examples"""
    totals = defaultdict(int)
    totals["total_examples"] = len(all_metrics)

    # Count metrics
    for m in all_metrics:
        totals["exact_match"] += int(m["exact_match"])
        totals["json_valid"] += int(m["json_valid"])
        totals["tool_name_match"] += int(m.get("tool_name_match", False))
        totals["query_match"] += int(m.get("query_match", False))
        totals["args_exact_match"] += int(m.get("args_exact_match", False))
        totals["args_keys_match"] += int(m.get("args_keys_match", False))
        totals["has_required_fields"] += int(m.get("has_required_fields", False))
        totals["args_partial_match_sum"] += m.get("args_partial_match", 0.0)

    # Calculate rates
    n = totals["total_examples"]
    results = {
        "total_examples": n,
        "exact_match_accuracy": totals["exact_match"] / n,
        "json_validity_rate": totals["json_valid"] / n,
        "tool_name_accuracy": totals["tool_name_match"] / n,
        "query_accuracy": totals["query_match"] / n,
        "args_exact_match_rate": totals["args_exact_match"] / n,
        "args_keys_match_rate": totals["args_keys_match"] / n,
        "args_partial_match_avg": totals["args_partial_match_sum"] / n,
        "has_required_fields_rate": totals["has_required_fields"] / n,
    }

    # Calculate composite scores
    # Functional correctness: tool name + args keys match (even if values differ slightly)
    results["functional_correctness"] = (
        totals["tool_name_match"] / n * 0.5 +
        totals["args_keys_match"] / n * 0.5
    )

    # Semantic correctness: includes query preservation
    results["semantic_correctness"] = (
        totals["tool_name_match"] / n * 0.4 +
        totals["args_partial_match_sum"] / n * 0.4 +
        totals["query_match"] / n * 0.2
    )

    return results

def main():
    # Load model
    model, tokenizer = load_model()

    # Load test data
    print(f"\nLoading test data from {TEST_FILE}...")
    examples = []
    with open(TEST_FILE) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Evaluating {len(examples)} examples...\n")

    # Evaluate each example
    all_metrics = []
    detailed_results = []

    for i, example in enumerate(examples, 1):
        prompt = example["prompt"]
        expected_str = example["completion"]
        expected = json.loads(expected_str)

        # Generate prediction
        prediction_str = generate_prediction(model, tokenizer, prompt)

        # Parse prediction
        try:
            predicted = json.loads(prediction_str)
        except json.JSONDecodeError:
            predicted = {"error": "invalid_json", "raw": prediction_str}

        # Evaluate
        metrics = evaluate_example(expected, predicted)
        all_metrics.append(metrics)

        # Store detailed result
        detailed_results.append({
            "example_id": i,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "expected": expected,
            "predicted": predicted,
            "metrics": metrics
        })

        # Progress
        if i % 10 == 0:
            print(f"Processed {i}/{len(examples)} examples...")

    # Aggregate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    aggregated = aggregate_metrics(all_metrics)

    print(f"\n📊 Overall Metrics:")
    print(f"  Total Examples: {aggregated['total_examples']}")
    print(f"\n🎯 Core Accuracy:")
    print(f"  Exact Match:        {aggregated['exact_match_accuracy']:.1%}")
    print(f"  Tool Name:          {aggregated['tool_name_accuracy']:.1%}")
    print(f"  Query Preservation: {aggregated['query_accuracy']:.1%}")
    print(f"\n🔧 Arguments Quality:")
    print(f"  Exact Match:        {aggregated['args_exact_match_rate']:.1%}")
    print(f"  Keys Match:         {aggregated['args_keys_match_rate']:.1%}")
    print(f"  Partial Match:      {aggregated['args_partial_match_avg']:.1%}")
    print(f"\n✅ Validity Checks:")
    print(f"  JSON Validity:      {aggregated['json_validity_rate']:.1%}")
    print(f"  Has Required Fields:{aggregated['has_required_fields_rate']:.1%}")
    print(f"\n🎓 Composite Scores:")
    print(f"  Functional Correctness: {aggregated['functional_correctness']:.1%}")
    print(f"  Semantic Correctness:   {aggregated['semantic_correctness']:.1%}")

    # Save detailed results
    output_file = "eval_structured_detailed_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": aggregated,
            "detailed_results": detailed_results
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Show some examples
    print(f"\n" + "="*80)
    print("SAMPLE CORRECT PREDICTIONS (first 3)")
    print("="*80)

    correct_examples = [r for r in detailed_results if r["metrics"]["exact_match"]][:3]
    for ex in correct_examples:
        print(f"\n✅ Example {ex['example_id']}:")
        print(f"   Prompt: {ex['prompt']}")
        print(f"   Tool: {ex['predicted'].get('tool_name', 'N/A')}")
        print(f"   Args: {ex['predicted'].get('arguments', {})}")

    print(f"\n" + "="*80)
    print("SAMPLE PARTIAL MATCHES (first 3)")
    print("="*80)

    partial_examples = [
        r for r in detailed_results
        if not r["metrics"]["exact_match"]
        and r["metrics"].get("tool_name_match", False)
        and r["metrics"].get("args_partial_match", 0) > 0.5
    ][:3]

    for ex in partial_examples:
        print(f"\n⚠️  Example {ex['example_id']}:")
        print(f"   Prompt: {ex['prompt']}")
        print(f"   Expected Args: {ex['expected'].get('arguments', {})}")
        print(f"   Predicted Args: {ex['predicted'].get('arguments', {})}")
        print(f"   Partial Match: {ex['metrics'].get('args_partial_match', 0):.1%}")

    print("\n" + "="*80)

    return aggregated

if __name__ == "__main__":
    results = main()
