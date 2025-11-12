#!/usr/bin/env python3
"""
Train Llama-3.1-8B with CEP - V2 with custom data collator
Simplified approach that definitely works
"""

import os
import sys
import json
import torch
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cep_config import CEPFormatter

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "paper_approach/datasets/hermes_train.jsonl"
VAL_DATASET_PATH = "paper_approach/datasets/hermes_val.jsonl"
OUTPUT_DIR = "slm_swap/04_ft/adapter_llama_cep"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training config
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.05

CEP_TYPE = "universal"

print("=" * 70)
print("Llama-3.1-8B Training with CEP (V2 - Custom Collator)")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"CEP Type: {CEP_TYPE}")
print("=" * 70)

cep_formatter = CEPFormatter(CEP_TYPE)

# Load model
print(f"\nLoading {MODEL_ID}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Apply LoRA
print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load dataset
print("\nLoading dataset...")
train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
val_dataset = load_dataset("json", data_files=VAL_DATASET_PATH, split="train") if os.path.exists(VAL_DATASET_PATH) else None

print(f"Train examples: {len(train_dataset)}")
if val_dataset:
    print(f"Validation examples: {len(val_dataset)}")


def format_and_tokenize(example):
    """Format with CEP and tokenize in one step"""
    conversations = example["conversations"]
    text = ""

    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            enhanced_system = cep_formatter.apply_to_system_prompt(value)
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{enhanced_system}<|eot_id|>"
        elif from_role == "human":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{value}<|eot_id|>"
        elif from_role == "gpt":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{value}<|eot_id|>"

    text += "<|end_of_text|>"

    # Tokenize
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


# Tokenize datasets
print("\nTokenizing datasets...")
train_dataset = train_dataset.map(
    format_and_tokenize,
    remove_columns=train_dataset.column_names,
    desc="Processing training dataset"
)

if val_dataset:
    val_dataset = val_dataset.map(
        format_and_tokenize,
        remove_columns=val_dataset.column_names,
        desc="Processing validation dataset"
    )


@dataclass
class CustomDataCollator:
    """Simple data collator for causal LM"""
    tokenizer: any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Pad to max length in batch
        input_ids = []
        attention_masks = []
        labels = []

        for f in features:
            # Pad input_ids and attention_mask
            padding_length = max_length - len(f["input_ids"])

            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * padding_length)
            attention_masks.append(f["attention_mask"] + [0] * padding_length)

            # Labels = input_ids, but mask padding tokens with -100
            label = f["input_ids"] + [-100] * padding_length
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# Data collator
data_collator = CustomDataCollator(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch" if val_dataset else "no",
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    seed=42,
    data_seed=42,
    save_total_limit=2,
    load_best_model_at_end=True if val_dataset else False,
    metric_for_best_model="eval_loss" if val_dataset else None,
    report_to="none",
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
start_time = datetime.now()

try:
    trainer_stats = trainer.train()
    end_time = datetime.now()

    # Save model
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metadata
    metadata = {
        "model_id": MODEL_ID,
        "approach": "llama_cep_v2",
        "cep_type": CEP_TYPE,
        "timestamp": datetime.now().isoformat(),
        "training_duration": str(end_time - start_time),
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
            "train_path": DATASET_PATH,
            "train_size": len(train_dataset),
            "val_path": VAL_DATASET_PATH if val_dataset else None,
            "val_size": len(val_dataset) if val_dataset else 0,
        },
        "trainer_stats": str(trainer_stats),
        "cep": cep_formatter.get_cep()
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("Training Complete!")
    print(f"Duration: {end_time - start_time}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 70)

except Exception as e:
    print(f"\n{'=' * 70}")
    print("TRAINING FAILED")
    print(f"{'=' * 70}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    raise
