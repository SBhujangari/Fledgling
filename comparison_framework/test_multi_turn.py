#!/usr/bin/env python3
"""
Multi-turn conversation test suite
Tests complex interactions with multiple function calls in conversation context
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset


class MultiTurnTester:
    """Test multi-turn conversation capabilities"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.examples = self.load_multi_turn_examples()

    def load_multi_turn_examples(self) -> List[Dict]:
        """Load only multi-turn examples from dataset"""
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

            # Multi-turn: more than 1 user message
            if len(user_messages) > 1:
                examples.append(example)

        print(f"Found {len(examples)} multi-turn examples out of {len(dataset)} total")
        return examples

    def categorize_by_conversation_complexity(self) -> Dict[str, List[Dict]]:
        """Categorize multi-turn examples by conversation complexity"""
        categories = {
            "clarification": [],     # User asks for clarification/refinement
            "sequential": [],        # Sequential tool calls (step 1, then step 2)
            "context_dependent": [], # Later calls depend on earlier context
        }

        for example in self.examples:
            messages = example.get("messages", [])
            user_messages = [m for m in messages if m.get("role") == "user"]

            # Heuristics for categorization
            # Clarification: user messages contain words like "no", "actually", "instead"
            clarification_keywords = ["no", "actually", "instead", "wait", "correction"]
            has_clarification = any(
                any(kw in msg.get("content", "").lower() for kw in clarification_keywords)
                for msg in user_messages
            )

            # Sequential: multiple distinct tool calls
            # (In Hermes, this might be represented as multiple function calls)
            # For now, use turn count as proxy
            turn_count = len(user_messages)

            if has_clarification:
                categories["clarification"].append(example)
            elif turn_count >= 3:
                categories["context_dependent"].append(example)
            else:
                categories["sequential"].append(example)

        return categories

    def create_test_splits(self, clarification_n: int = 10, sequential_n: int = 10, context_n: int = 10):
        """Create balanced test split across conversation types"""
        categorized = self.categorize_by_conversation_complexity()

        print("\nMulti-turn categorization:")
        for category, items in categorized.items():
            print(f"  {category}: {len(items)} examples")

        test_split = {
            "clarification": categorized["clarification"][:clarification_n],
            "sequential": categorized["sequential"][:sequential_n],
            "context_dependent": categorized["context_dependent"][:context_n],
        }

        # Flatten for evaluation
        all_test = []
        for category, items in test_split.items():
            for item in items:
                item["_conversation_type"] = category
                all_test.append(item)

        return all_test

    def save_test_split(self, output_path: str, clarification_n: int = 10, sequential_n: int = 10, context_n: int = 10):
        """Save multi-turn test split"""
        test_examples = self.create_test_splits(clarification_n, sequential_n, context_n)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in test_examples:
                f.write(json.dumps(example) + "\n")

        print(f"\nSaved {len(test_examples)} multi-turn test examples to: {output_path}")

        # Save summary
        summary = {
            "total": len(test_examples),
            "clarification": clarification_n,
            "sequential": sequential_n,
            "context_dependent": context_n,
        }

        summary_path = output_path.parent / "multi_turn_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_path}")

        return test_examples


def main():
    parser = argparse.ArgumentParser(description="Create multi-turn test split")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to full dataset (JSONL)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output path for multi-turn test split")
    parser.add_argument("--clarification", type=int, default=10,
                        help="Number of clarification examples")
    parser.add_argument("--sequential", type=int, default=10,
                        help="Number of sequential examples")
    parser.add_argument("--context", type=int, default=10,
                        help="Number of context-dependent examples")

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Turn Conversation Test Suite")
    print("=" * 60)

    tester = MultiTurnTester(args.dataset_path)

    if len(tester.examples) == 0:
        print("\nWARNING: No multi-turn examples found in dataset!")
        print("This dataset may only contain single-turn examples.")
        print("Skipping multi-turn test creation.")
        return

    tester.save_test_split(args.output_path, args.clarification, args.sequential, args.context)

    print("=" * 60)
    print("Multi-turn test split created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
