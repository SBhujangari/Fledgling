#!/usr/bin/env python3
"""
Training script for function calling following FUNCTION_CALLING_FINETUNE.md approach
Supports both Phi-4-mini and Llama-3.1-8B-Instruct
"""

import os
import json
import argparse
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import torch

# Set deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def format_hermes_example(example):
    """
    Format Hermes Function Calling dataset example
    Actual Hermes format:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    The tool schema is embedded in the system message.
    The assistant response includes the tool call.
    """
    conversations = example.get("conversations", [])

    # Build full conversation text
    text = ""
    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            text += f"<|system|>\n{value}\n\n"
        elif from_role == "human":
            text += f"<|user|>\n{value}\n\n"
        elif from_role == "gpt":
            text += f"<|assistant|>\n{value}\n"

    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Train function calling model (Paper approach)")
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "microsoft/phi-4-mini"),
                        help="Base model ID (default: microsoft/phi-4-mini)")
    parser.add_argument("--dataset-path", type=str, default="datasets/hermes_function_calling_train.jsonl",
                        help="Path to Hermes training dataset")
    parser.add_argument("--val-dataset-path", type=str, default="datasets/hermes_function_calling_val.jsonl",
                        help="Path to Hermes validation dataset")
    parser.add_argument("--output-dir", type=str, default="adapters/lora_out",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")

    args = parser.parse_args()

    print("=" * 60)
    print("FUNCTION_CALLING_FINETUNE - Training Start")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Training: BS={args.batch_size}, AccumSteps={args.grad_accum}, LR={args.lr}, Epochs={args.epochs}")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset("json", data_files=args.dataset_path, split="train")
    val_ds = None
    if os.path.exists(args.val_dataset_path):
        val_ds = load_dataset("json", data_files=args.val_dataset_path, split="train")

    # Format datasets
    print("Formatting examples...")
    train_ds = train_ds.map(format_hermes_example, remove_columns=train_ds.column_names)
    if val_ds:
        val_ds = val_ds.map(format_hermes_example, remove_columns=val_ds.column_names)

    print(f"Train examples: {len(train_ds)}")
    if val_ds:
        print(f"Val examples: {len(val_ds)}")

    # Load model with Unsloth
    print(f"\nLoading base model: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        use_gradient_checkpointing=True,
    )

    # Apply LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        bf16=True,
        optim="adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=42,
        data_seed=42,
    )

    # Trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )

    # Train
    print("\nStarting training...")
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    # Save
    print("\nSaving model and tokenizer...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "timestamp": datetime.now().isoformat(),
        "training_duration": str(end_time - start_time),
        "hyperparams": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
        },
        "dataset": {
            "train_path": args.dataset_path,
            "train_size": len(train_ds),
            "val_path": args.val_dataset_path if val_ds else None,
            "val_size": len(val_ds) if val_ds else 0,
        },
        "approach": "FUNCTION_CALLING_FINETUNE_paper",
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("Training complete!")
    print(f"Duration: {end_time - start_time}")
    print(f"Adapters saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
