"""One-off sanity inference for either track and client."""

from __future__ import annotations

import argparse
import json
import os

from clients import build_client
from prompts import get_system_prompt


def load_first_prompt(dataset_path: str) -> str:
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                return obj["prompt"]
    raise ValueError(f"No records found in {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a single prompt through a selected client.")
    parser.add_argument("--track", choices=("structured", "toolcall"), required=True)
    parser.add_argument("--model-kind", choices=("slm", "azure"), required=True)
    parser.add_argument("--prompt", help="Prompt text; defaults to first train row.")
    parser.add_argument(
        "--dataset",
        help="Dataset to sample prompt from (defaults to track train split).",
    )
    parser.add_argument("--adapter", help="Optional LoRA adapter for SLM.")
    args = parser.parse_args()

    dataset_path = args.dataset or os.path.join(
        "02_dataset", args.track, "train.jsonl"
    )
    prompt = args.prompt or load_first_prompt(dataset_path)
    system_prompt = get_system_prompt(args.track)
    client = build_client(args.model_kind, adapter=args.adapter)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.generate(messages)
    print(response)


if __name__ == "__main__":
    main()
