#!/usr/bin/env python3
"""
Train Llama-3.1-8B with Context Engineering Prefix (CEP)

Approach:
1. Universal CEP applied to ALL training examples
2. Enforces rigid structure for both JSON and XML outputs
3. Simple, generalizable, effective
"""

import os
import sys
import json
import torch
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Import CEP
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cep_config import CEPFormatter

# Set deterministic
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "paper_approach/datasets/hermes_train.jsonl"
VAL_DATASET_PATH = "paper_approach/datasets/hermes_val.jsonl"
OUTPUT_DIR = "slm_swap/04_ft/adapter_llama_cep"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config optimized for Llama
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training config
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.05

# CEP Configuration
CEP_TYPE = "universal"  # or "compact" for shorter prefix

print("=" * 70)
print("Llama-3.1-8B Training with Context Engineering Prefix (CEP)")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Training: BS={BATCH_SIZE}, AccumSteps={GRAD_ACCUM_STEPS}, LR={LEARNING_RATE}")
print(f"CEP Type: {CEP_TYPE}")
print("=" * 70)

# Initialize CEP
cep_formatter = CEPFormatter(CEP_TYPE)
print("\nContext Engineering Prefix:")
print(cep_formatter.get_cep()[:200] + "...")

# Load model
print(f"\nLoading {MODEL_ID}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply LoRA
print("Applying LoRA adapters...")
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
print("\nLoading Hermes dataset...")
train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
val_dataset = load_dataset("json", data_files=VAL_DATASET_PATH, split="train") if os.path.exists(VAL_DATASET_PATH) else None

print(f"Train examples: {len(train_dataset)}")
if val_dataset:
    print(f"Validation examples: {len(val_dataset)}")


def format_hermes_with_cep(examples):
    """
    Format Hermes examples with CEP applied to system message

    Llama-3.1-Instruct format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    [CEP + Original System Message]<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    [User Message]<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    [Assistant Response]<|eot_id|><|end_of_text|>
    """
    texts = []

    # Handle both single example and batch
    conversations_list = examples["conversations"] if isinstance(examples["conversations"][0], list) else [examples["conversations"]]

    for conversations in conversations_list:
        text = ""

        for msg in conversations:
            from_role = msg.get("from", "")
            value = msg.get("value", "")

            if from_role == "system":
                # Apply CEP to system message
                enhanced_system = cep_formatter.apply_to_system_prompt(value)
                text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{enhanced_system}<|eot_id|>"

            elif from_role == "human":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{value}<|eot_id|>"

            elif from_role == "gpt":
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{value}<|eot_id|>"

        text += "<|end_of_text|>"
        texts.append(text)

    return texts


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
)

# Trainer
print("\nInitializing SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    formatting_func=format_hermes_with_cep,
    packing=False,
    args=training_args,
)

# Train
print("\nStarting training...")
start_time = datetime.now()
trainer_stats = trainer.train()
end_time = datetime.now()

# Save model
print("\nSaving model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
metadata = {
    "model_id": MODEL_ID,
    "approach": "llama_cep",
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
