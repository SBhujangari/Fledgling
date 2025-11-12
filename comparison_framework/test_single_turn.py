#!/usr/bin/env python3
"""
Single-turn function calling test suite
Tests direct query -> function call mappings
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset


class SingleTurnTester:
    """Test single-turn function calling capabilities"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.examples = self.load_single_turn_examples()

    def load_single_turn_examples(self) -> List[Dict]:
        """Load only single-turn examples from dataset"""
        print(f"Loading dataset from: {self.dataset_path}")

        if not self.dataset_path.exists():
            print(f"WARNING: Dataset not found at {self.dataset_path}")
            return []

        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        examples = []

        for example in dataset:
            messages = example.get("messages", [])
            # Count user messages
            user_messages = [m for m in messages if m.get("role") == "user"]

            # Single-turn: exactly 1 user message
            if len(user_messages) == 1:
                examples.append(example)

        print(f"Found {len(examples)} single-turn examples out of {len(dataset)} total")
        return examples

    def categorize_by_difficulty(self) -> Dict[str, List[Dict]]:
        """Categorize single-turn examples by difficulty"""
        categories = {
            "simple": [],      # 1 tool, required params only
            "moderate": [],    # 1 tool, optional params
            "complex": [],     # Multiple tools or nested params
        }

        for example in self.examples:
            tools = example.get("tools", [])

            if len(tools) == 0:
                continue
            elif len(tools) == 1:
                # Check parameter complexity
                tool = tools[0]
                params = tool.get("parameters", {})
                properties = params.get("properties", {})
                required = params.get("required", [])

                # Simple: only required params, no nesting
                has_nested = any(
                    prop.get("type") in ["object", "array"]
                    for prop in properties.values()
                )

                if not has_nested and len(required) == len(properties):
                    categories["simple"].append(example)
                elif not has_nested:
                    categories["moderate"].append(example)
                else:
                    categories["complex"].append(example)
            else:
                # Multiple tools = complex
                categories["complex"].append(example)

        return categories

    def create_test_splits(self, simple_n: int = 20, moderate_n: int = 20, complex_n: int = 10):
        """Create balanced test split across difficulty levels"""
        categorized = self.categorize_by_difficulty()

        print("\nSingle-turn categorization:")
        for category, items in categorized.items():
            print(f"  {category}: {len(items)} examples")

        test_split = {
            "simple": categorized["simple"][:simple_n],
            "moderate": categorized["moderate"][:moderate_n],
            "complex": categorized["complex"][:complex_n],
        }

        # Flatten for evaluation
        all_test = []
        for category, items in test_split.items():
            for item in items:
                item["_difficulty"] = category
                all_test.append(item)

        return all_test

    def save_test_split(self, output_path: str, simple_n: int = 20, moderate_n: int = 20, complex_n: int = 10):
        """Save single-turn test split"""
        test_examples = self.create_test_splits(simple_n, moderate_n, complex_n)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in test_examples:
                f.write(json.dumps(example) + "\n")

        print(f"\nSaved {len(test_examples)} single-turn test examples to: {output_path}")

        # Save summary
        summary = {
            "total": len(test_examples),
            "simple": simple_n,
            "moderate": moderate_n,
            "complex": complex_n,
        }

        summary_path = output_path.parent / "single_turn_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_path}")

        return test_examples


def main():
    parser = argparse.ArgumentParser(description="Create single-turn test split")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to full dataset (JSONL)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output path for single-turn test split")
    parser.add_argument("--simple", type=int, default=20,
                        help="Number of simple examples")
    parser.add_argument("--moderate", type=int, default=20,
                        help="Number of moderate examples")
    parser.add_argument("--complex", type=int, default=10,
                        help="Number of complex examples")

    args = parser.parse_args()

    print("=" * 60)
    print("Single-Turn Function Calling Test Suite")
    print("=" * 60)

    tester = SingleTurnTester(args.dataset_path)
    tester.save_test_split(args.output_path, args.simple, args.moderate, args.complex)

    print("=" * 60)
    print("Single-turn test split created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
