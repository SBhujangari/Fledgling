#!/usr/bin/env python3
"""
Download and prepare Hermes Function Calling v1 dataset
Splits into train/val/test for iterative experimentation
"""

import os
import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict


def categorize_by_complexity(examples):
    """
    Categorize examples by complexity:
    - Single-turn: 1 user message, direct function call
    - Multi-turn: Multiple messages in conversation
    - Simple domains: Basic operations (weather, calendar, search)
    - Complex domains: Multi-step workflows, nested data
    """
    categorized = {
        "single_turn": [],
        "multi_turn": [],
        "simple_domain": [],
        "complex_domain": [],
    }

    for example in examples:
        messages = example.get("messages", [])
        tools = example.get("tools", [])

        # Count user messages
        user_messages = [m for m in messages if m.get("role") == "user"]

        if len(user_messages) == 1:
            categorized["single_turn"].append(example)
        else:
            categorized["multi_turn"].append(example)

        # Simple heuristic for domain complexity
        # Simple: 1 tool, simple parameters
        # Complex: Multiple tools or nested parameters
        if len(tools) == 1:
            params = tools[0].get("parameters", {})
            properties = params.get("properties", {})
            # Check if any property is an object or array
            has_complex = any(
                prop.get("type") in ["object", "array"]
                for prop in properties.values()
            )
            if not has_complex:
                categorized["simple_domain"].append(example)
            else:
                categorized["complex_domain"].append(example)
        else:
            categorized["complex_domain"].append(example)

    return categorized


def main():
    parser = argparse.ArgumentParser(description="Prepare Hermes Function Calling dataset")
    parser.add_argument("--dataset-name", type=str,
                        default="NousResearch/hermes-function-calling-v1",
                        help="Hugging Face dataset name")
    parser.add_argument("--output-dir", type=str, default="datasets",
                        help="Output directory")
    parser.add_argument("--train-size", type=int, default=100,
                        help="Training examples (for Phase 1 iteration)")
    parser.add_argument("--val-size", type=int, default=50,
                        help="Validation examples")
    parser.add_argument("--test-size", type=int, default=50,
                        help="Test examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stratify", action="store_true",
                        help="Stratify by complexity")

    args = parser.parse_args()

    print("=" * 60)
    print("Hermes Function Calling Dataset Preparation")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"Splits: Train={args.train_size}, Val={args.val_size}, Test={args.test_size}")
    print(f"Stratify: {args.stratify}")
    print("=" * 60)

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nDownloading dataset from Hugging Face...")
    try:
        dataset = load_dataset(args.dataset_name, split="train")
        print(f"Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("\nTrying alternative dataset name: NousResearch/hermes-2-pro-function-calling")
        try:
            dataset = load_dataset("NousResearch/hermes-2-pro-function-calling", split="train")
            print(f"Loaded {len(dataset)} examples from alternative source")
        except Exception as e2:
            print(f"ERROR: Alternative also failed: {e2}")
            print("\nFalling back to creating synthetic examples...")
            dataset = create_synthetic_hermes_examples()

    # Convert to list
    examples = list(dataset)

    # Shuffle
    random.shuffle(examples)

    # Stratify if requested
    if args.stratify:
        print("\nCategorizing by complexity...")
        categorized = categorize_by_complexity(examples)

        print("\nCategory distribution:")
        for category, items in categorized.items():
            print(f"  {category}: {len(items)} examples")

        # Sample proportionally from each category
        train_examples = []
        val_examples = []
        test_examples = []

        for category, items in categorized.items():
            random.shuffle(items)
            # Proportional split
            n = len(items)
            total = args.train_size + args.val_size + args.test_size
            train_n = int(n * args.train_size / total)
            val_n = int(n * args.val_size / total)
            test_n = int(n * args.test_size / total)

            train_examples.extend(items[:train_n])
            val_examples.extend(items[train_n:train_n+val_n])
            test_examples.extend(items[train_n+val_n:train_n+val_n+test_n])

        # Shuffle again
        random.shuffle(train_examples)
        random.shuffle(val_examples)
        random.shuffle(test_examples)

    else:
        # Simple split
        train_examples = examples[:args.train_size]
        val_examples = examples[args.train_size:args.train_size+args.val_size]
        test_examples = examples[args.train_size+args.val_size:args.train_size+args.val_size+args.test_size]

    # Save splits
    print("\nSaving splits...")
    splits = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    for split_name, split_examples in splits.items():
        output_path = output_dir / f"hermes_{split_name}.jsonl"
        with open(output_path, "w") as f:
            for example in split_examples:
                f.write(json.dumps(example) + "\n")
        print(f"  {split_name}: {len(split_examples)} examples -> {output_path}")

    # Generate statistics
    print("\nGenerating statistics...")
    stats = {
        "dataset_name": args.dataset_name,
        "total_examples": len(examples),
        "splits": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "seed": args.seed,
        "stratified": args.stratify,
    }

    if args.stratify:
        stats["category_distribution"] = {
            category: len(items)
            for category, items in categorized.items()
        }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {stats_path}")
    print("=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


def create_synthetic_hermes_examples():
    """Create synthetic examples if dataset download fails"""
    print("Creating synthetic Hermes-format examples...")

    examples = []

    # Example 1: Simple weather query
    examples.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to weather tools."},
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "description": "Temperature units (celsius/fahrenheit)"}
                },
                "required": ["location"]
            }
        }],
        "answer": {
            "tool_call": {
                "name": "get_weather",
                "arguments": {
                    "location": "San Francisco",
                    "units": "fahrenheit"
                }
            }
        }
    })

    # Example 2: Search query
    examples.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to search tools."},
            {"role": "user", "content": "Search for Python tutorials for beginners"}
        ],
        "tools": [{
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }],
        "answer": {
            "tool_call": {
                "name": "web_search",
                "arguments": {
                    "query": "Python tutorials for beginners",
                    "num_results": 10
                }
            }
        }
    })

    # Replicate these examples to reach 200 total (100 train + 50 val + 50 test)
    base_examples = examples.copy()
    while len(examples) < 200:
        examples.extend(base_examples)

    examples = examples[:200]

    print(f"Created {len(examples)} synthetic examples")

    return examples


if __name__ == "__main__":
    main()
