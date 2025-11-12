#!/usr/bin/env python3
"""
Simplified training script using Unsloth's recommended approach
"""

import os
import json
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import torch

# Set deterministic
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/phi-4",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=42,
)

# Load dataset
dataset = load_dataset("json", data_files="paper_approach/datasets/hermes_train.jsonl", split="train")

def formatting_func(examples):
    texts = []
    for i in range(len(examples["conversations"])):
        conversations = examples["conversations"][i]
        text = ""
        for msg in conversations:
            from_role = msg.get("from", "")
            value = msg.get("value", "")
            if from_role == "system":
                text += f"<|system|>\n{value}\n\n"
            elif from_role == "human":
                text += f"<|user|>\n{value}\n\n"
            elif from_role == "gpt":
                text += f"<|assistant|>\n{value}"
        texts.append(text)
    return texts

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    formatting_func=formatting_func,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="paper_approach/adapters/phi_hermes_lora",
        save_strategy="epoch",
    ),
)

# Train
trainer_stats = trainer.train()

# Save
model.save_pretrained("paper_approach/adapters/phi_hermes_lora")
tokenizer.save_pretrained("paper_approach/adapters/phi_hermes_lora")

print(f"Training complete! Stats: {trainer_stats}")
