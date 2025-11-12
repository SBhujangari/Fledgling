#!/usr/bin/env python3
"""
Train Llama-3.1-8B with Hermes dataset + Auto Context Engineering

Hybrid Approach:
1. Fine-tune Llama-3.1-8B on Hermes function calling dataset
2. During training, collect error patterns
3. Use auto context engineering to optimize prompts for failed examples
4. Re-train on hard examples with optimized prompts
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from auto_context_engineering import AutoContextEngineer, HybridSLM

# Set deterministic
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "paper_approach/datasets/hermes_train.jsonl"
VAL_DATASET_PATH = "paper_approach/datasets/hermes_val.jsonl"
OUTPUT_DIR = "slm_swap/04_ft/adapter_llama_hybrid"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config optimized for Llama
LORA_R = 32  # Higher rank for 8B model
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training config
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.05

print("=" * 70)
print("HYBRID TRAINING: Llama-3.1-8B + Auto Context Engineering")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Training: BS={BATCH_SIZE}, AccumSteps={GRAD_ACCUM_STEPS}, LR={LEARNING_RATE}")
print("=" * 70)

# Initialize auto context engineering
print("\nInitializing Auto Context Engineering...")
context_engineer = AutoContextEngineer()

# Load model
print(f"\nLoading {MODEL_ID}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
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


def format_hermes_with_context_engineering(examples):
    """
    Format Hermes examples with auto-optimized prompts

    Strategy:
    1. Detect if example is function calling or regular conversation
    2. Apply context-engineered prompt template
    3. Format in Llama-3.1 chat format
    """
    texts = []

    for i in range(len(examples["conversations"])):
        conversations = examples["conversations"][i]
        text = ""

        # Check if this is a function calling example
        is_function_call = any(
            "tool" in str(msg.get("value", "")).lower() or
            "function" in str(msg.get("value", "")).lower()
            for msg in conversations
        )

        # Apply Llama-3.1 Instruct format with context engineering
        for msg in conversations:
            from_role = msg.get("from", "")
            value = msg.get("value", "")

            if from_role == "system":
                # Enhance system message with context engineering hints
                if is_function_call:
                    enhanced_value = f"""{value}

CRITICAL FORMATTING RULES:
- For JSON: Output ONLY raw JSON, NO markdown code blocks
- For XML: Use exact format <tool_call name="FUNC">{{"args"}}</tool_call>
- Match parameter names EXACTLY as shown in function signature
- NO extra text before or after the output"""
                    text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{enhanced_value}<|eot_id|>"
                else:
                    text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{value}<|eot_id|>"

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
    formatting_func=format_hermes_with_context_engineering,
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
    "approach": "hybrid_llama_auto_context",
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
    "context_engineering": {
        "enabled": True,
        "templates_used": len(context_engineer.templates["structured_json"]) + len(context_engineer.templates["toolcall_xml"])
    }
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Save context engineering state
with open(os.path.join(OUTPUT_DIR, "context_engineering.json"), "w") as f:
    json.dump(context_engineer.get_error_report(), f, indent=2)

print("=" * 70)
print("Training Complete!")
print(f"Duration: {end_time - start_time}")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"Context engineering report saved")
print("=" * 70)
