#!/usr/bin/env python3
"""
Convert slm_swap structured dataset to Llama training format
Ensures eval and train use EXACT same format for guaranteed match
"""

import json
from pathlib import Path

def convert_to_chat_format(example):
    """Convert slm_swap structured format to Llama chat format"""
    prompt = example["prompt"]
    completion = example["completion"]

    # Create chat format matching training script expectations
    # Training script expects "conversations" with "from" and "value" keys
    return {
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": completion
            }
        ]
    }

def main():
    # Paths
    train_in = Path("slm_swap/02_dataset/structured/train.jsonl")
    val_in = Path("slm_swap/02_dataset/structured/val.jsonl")

    train_out = Path("slm_swap/02_dataset/structured/train_chat.jsonl")
    val_out = Path("slm_swap/02_dataset/structured/val_chat.jsonl")

    # Convert training data
    print(f"Converting {train_in}...")
    with open(train_in) as f_in, open(train_out, "w") as f_out:
        count = 0
        for line in f_in:
            example = json.loads(line)
            converted = convert_to_chat_format(example)
            f_out.write(json.dumps(converted) + "\n")
            count += 1
        print(f"  ✓ Converted {count} training examples")

    # Convert validation data
    print(f"Converting {val_in}...")
    with open(val_in) as f_in, open(val_out, "w") as f_out:
        count = 0
        for line in f_in:
            example = json.loads(line)
            converted = convert_to_chat_format(example)
            f_out.write(json.dumps(converted) + "\n")
            count += 1
        print(f"  ✓ Converted {count} validation examples")

    print(f"\n✅ Conversion complete!")
    print(f"Training: {train_out}")
    print(f"Validation: {val_out}")

if __name__ == "__main__":
    main()
