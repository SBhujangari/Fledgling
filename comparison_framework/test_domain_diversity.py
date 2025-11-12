#!/usr/bin/env python3
"""
Domain diversity test suite
Tests performance across diverse domains (weather, e-commerce, search, etc.)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from collections import defaultdict


class DomainDiversityTester:
    """Test domain diversity capabilities"""

    # Common domain keywords
    DOMAIN_KEYWORDS = {
        "weather": ["weather", "temperature", "forecast", "climate"],
        "search": ["search", "query", "find", "lookup"],
        "ecommerce": ["order", "purchase", "cart", "product", "shop"],
        "calendar": ["calendar", "event", "schedule", "meeting", "appointment"],
        "maps": ["location", "map", "navigate", "route", "directions"],
        "communication": ["email", "message", "send", "call", "contact"],
        "finance": ["payment", "transaction", "balance", "account", "transfer"],
        "database": ["database", "record", "query", "table", "insert"],
        "files": ["file", "document", "upload", "download", "storage"],
        "social": ["post", "tweet", "share", "like", "follow"],
    }

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.examples = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        """Load dataset"""
        print(f"Loading dataset from: {self.dataset_path}")

        if not self.dataset_path.exists():
            print(f"WARNING: Dataset not found at {self.dataset_path}")
            return []

        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        return list(dataset)

    def classify_domain(self, example: Dict) -> str:
        """Classify an example into a domain"""
        # Extract text from messages and tools
        messages = example.get("messages", [])
        tools = example.get("tools", [])

        # Combine all text
        text_parts = []
        for msg in messages:
            text_parts.append(msg.get("content", "").lower())

        for tool in tools:
            text_parts.append(tool.get("name", "").lower())
            text_parts.append(tool.get("description", "").lower())

        combined_text = " ".join(text_parts)

        # Match against domain keywords
        domain_scores = defaultdict(int)
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    domain_scores[domain] += 1

        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "other"

    def categorize_by_domain(self) -> Dict[str, List[Dict]]:
        """Categorize examples by domain"""
        domains = defaultdict(list)

        for example in self.examples:
            domain = self.classify_domain(example)
            example["_domain"] = domain
            domains[domain].append(example)

        return dict(domains)

    def create_balanced_split(self, examples_per_domain: int = 5) -> List[Dict]:
        """Create balanced test split across domains"""
        categorized = self.categorize_by_domain()

        print("\nDomain distribution:")
        for domain, items in sorted(categorized.items(), key=lambda x: -len(x[1])):
            print(f"  {domain}: {len(items)} examples")

        # Sample from each domain
        balanced_test = []
        for domain, items in categorized.items():
            sampled = items[:examples_per_domain]
            balanced_test.extend(sampled)

        print(f"\nBalanced test set: {len(balanced_test)} examples across {len(categorized)} domains")

        return balanced_test

    def save_test_split(self, output_path: str, examples_per_domain: int = 5):
        """Save domain diversity test split"""
        test_examples = self.create_balanced_split(examples_per_domain)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in test_examples:
                f.write(json.dumps(example) + "\n")

        print(f"\nSaved {len(test_examples)} domain-diverse test examples to: {output_path}")

        # Save summary
        domain_counts = defaultdict(int)
        for example in test_examples:
            domain_counts[example.get("_domain", "unknown")] += 1

        summary = {
            "total": len(test_examples),
            "examples_per_domain": examples_per_domain,
            "domains": dict(domain_counts),
        }

        summary_path = output_path.parent / "domain_diversity_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_path}")

        return test_examples


def main():
    parser = argparse.ArgumentParser(description="Create domain diversity test split")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to full dataset (JSONL)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output path for domain diversity test split")
    parser.add_argument("--per-domain", type=int, default=5,
                        help="Number of examples per domain")

    args = parser.parse_args()

    print("=" * 60)
    print("Domain Diversity Test Suite")
    print("=" * 60)

    tester = DomainDiversityTester(args.dataset_path)

    if len(tester.examples) == 0:
        print("\nWARNING: No examples found in dataset!")
        return

    tester.save_test_split(args.output_path, args.per_domain)

    print("=" * 60)
    print("Domain diversity test split created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
