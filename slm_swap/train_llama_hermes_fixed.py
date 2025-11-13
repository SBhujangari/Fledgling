#!/usr/bin/env python3
"""
CORRECTED Hermes function calling training script.
Fixes the broken paper approach by using actual Llama 3.1 tokens instead of fake custom tokens.

Key fix: Uses <|begin_of_text|><|start_header_id|>...<|end_header_id|> instead of <|system|>
"""

import argparse
import os
import json
import torch
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List

# Set deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
DEFAULT_TRAIN = REPO_ROOT / "paper_approach" / "datasets" / "hermes_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "paper_approach" / "datasets" / "hermes_val.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "04_ft" / "adapter_llama_hermes_fixed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Llama-3.1-8B for Hermes function calling (CORRECTED)"
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HF repo id or local path for the base model.",
    )
    parser.add_argument(
        "--train",
        dest="train_dataset",
        default=str(DEFAULT_TRAIN),
        help="Path to Hermes-format training JSONL.",
    )
    parser.add_argument(
        "--val",
        dest="val_dataset",
        default=str(DEFAULT_VAL) if DEFAULT_VAL.exists() else "",
        help="Optional validation JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory where adapters + metadata are written.",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device train batch size."
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio.")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha.")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_ID = args.model_id
    DATASET_PATH = Path(args.train_dataset).expanduser().resolve()
    VAL_DATASET_PATH = Path(args.val_dataset).expanduser().resolve() if args.val_dataset else None
    OUTPUT_DIR = Path(args.output).expanduser().resolve()
    MAX_SEQ_LENGTH = args.max_length

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout

    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    WARMUP_RATIO = args.warmup_ratio

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATASET_PATH}")

    print("=" * 70)
    print("Llama-3.1-8B Hermes Function Calling Training (CORRECTED)")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("FIX: Using actual Llama tokens instead of fake <|system|> tokens")
    print("=" * 70)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    print(f"\nLoading {MODEL_ID} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare model for k-bit training
    print("Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\nLoading dataset...")
    train_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    val_dataset = None
    if VAL_DATASET_PATH and VAL_DATASET_PATH.exists():
        val_dataset = load_dataset("json", data_files=str(VAL_DATASET_PATH), split="train")

    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation examples: {len(val_dataset)}")

    def format_and_tokenize(example):
        """
        Format with CORRECT Llama 3.1 tokens.

        ❌ OLD (BROKEN): <|system|>, <|user|>, <|assistant|>
        ✅ NEW (CORRECT): <|start_header_id|>system<|end_header_id|>
        """
        conversations = example["conversations"]
        text = ""

        for msg in conversations:
            from_role = msg.get("from", "")
            value = msg.get("value", "")

            if from_role == "system":
                # ✅ CORRECT: Use actual Llama tokens
                text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{value}<|eot_id|>"
            elif from_role == "human":
                # ✅ CORRECT: Use actual Llama tokens
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{value}<|eot_id|>"
            elif from_role == "gpt":
                # ✅ CORRECT: Use actual Llama tokens
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{value}<|eot_id|>"

        text += "<|end_of_text|>"

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        format_and_tokenize,
        remove_columns=train_dataset.column_names,
        desc="Processing training dataset",
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            format_and_tokenize,
            remove_columns=val_dataset.column_names,
            desc="Processing validation dataset",
        )

    @dataclass
    class CustomDataCollator:
        """Data collator for causal LM"""

        tokenizer: any

        def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
            max_length = max(len(f["input_ids"]) for f in features)

            input_ids = []
            attention_masks = []
            labels = []

            for f in features:
                padding_length = max_length - len(f["input_ids"])

                input_ids.append(
                    f["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
                )
                attention_masks.append(f["attention_mask"] + [0] * padding_length)
                labels.append(f["input_ids"] + [-100] * padding_length)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        data_seed=42,
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    print("Expected: >50% args exact match (vs 64% Azure baseline)")
    start_time = datetime.now()

    try:
        trainer_stats = trainer.train()
        end_time = datetime.now()

        print("\nSaving model...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        metadata = {
            "model_id": MODEL_ID,
            "approach": "llama_hermes_FIXED",
            "timestamp": datetime.now().isoformat(),
            "training_duration": str(end_time - start_time),
            "fix_applied": "Used actual Llama tokens instead of fake <|system|> tokens",
            "hyperparams": {
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "max_seq_length": MAX_SEQ_LENGTH,
            },
            "dataset": {
                "train_path": str(DATASET_PATH),
                "train_size": len(train_dataset),
                "val_path": str(VAL_DATASET_PATH) if val_dataset else None,
                "val_size": len(val_dataset) if val_dataset else 0,
            },
            "trainer_stats": str(trainer_stats),
        }

        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print("=" * 70)
        print("Training Complete!")
        print(f"Duration: {end_time - start_time}")
        print(f"Model saved to: {OUTPUT_DIR}")
        print("=" * 70)
        print("\nNext step: Run evaluation to verify >0% accuracy")
        print(f"Expected: >50% args exact (vs 0% with broken script)")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print("TRAINING FAILED")
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
