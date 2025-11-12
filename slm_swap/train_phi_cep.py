#!/usr/bin/env python3
"""
Train Phi-4 with Context Engineering Prefix (CEP)

Lighter, faster model for sub-agents with CEP
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cep_config import CEPFormatter

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
MODEL_ID = "microsoft/phi-4"
DATASET_PATH = "paper_approach/datasets/hermes_train.jsonl"
VAL_DATASET_PATH = "paper_approach/datasets/hermes_val.jsonl"
OUTPUT_DIR = "slm_swap/04_ft/adapter_phi_cep"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config for Phi
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0  # Phi works better with 0 dropout

# Training config
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.05

CEP_TYPE = "compact"  # Use compact CEP for smaller model

print("=" * 70)
print("Phi-4 Training with Context Engineering Prefix (CEP)")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"CEP Type: {CEP_TYPE} (optimized for Phi)")
print("=" * 70)

cep_formatter = CEPFormatter(CEP_TYPE)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply LoRA
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
train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
val_dataset = load_dataset("json", data_files=VAL_DATASET_PATH, split="train") if os.path.exists(VAL_DATASET_PATH) else None

print(f"Train examples: {len(train_dataset)}")
if val_dataset:
    print(f"Validation examples: {len(val_dataset)}")


def format_hermes_with_cep(examples):
    """Format for Phi with CEP"""
    texts = []

    # Handle both single example and batch
    conversations_list = examples["conversations"] if isinstance(examples["conversations"][0], list) else [examples["conversations"]]

    for conversations in conversations_list:
        text = ""

        for msg in conversations:
            from_role = msg.get("from", "")
            value = msg.get("value", "")

            if from_role == "system":
                enhanced_system = cep_formatter.apply_to_system_prompt(value)
                text += f"<|system|>\n{enhanced_system}\n\n"

            elif from_role == "human":
                text += f"<|user|>\n{value}\n\n"

            elif from_role == "gpt":
                text += f"<|assistant|>\n{value}"

        texts.append(text)

    return texts


# Training
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
)

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

print("\nStarting training...")
start_time = datetime.now()
trainer_stats = trainer.train()
end_time = datetime.now()

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "model_id": MODEL_ID,
    "approach": "phi_cep",
    "cep_type": CEP_TYPE,
    "timestamp": datetime.now().isoformat(),
    "training_duration": str(end_time - start_time),
    "hyperparams": {
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    },
    "dataset": {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset) if val_dataset else 0,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("=" * 70)
print("Training Complete!")
print(f"Duration: {end_time - start_time}")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 70)
