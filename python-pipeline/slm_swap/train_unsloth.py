"""Conditional Unsloth QLoRA fine-tuning for the local SLM."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from contextlib import nullcontext

from datasets import Dataset
from langfuse_helpers import langfuse_trace
from prompts import get_system_prompt
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_dataset(dataset: Dataset, tokenizer, track: str) -> Dataset:
    system_prompt = get_system_prompt(track)

    def _format(batch):
        texts = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    return dataset.map(_format, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="Train Unsloth QLoRA adapters.")
    parser.add_argument("--track", choices=("structured", "toolcall"), required=True)
    parser.add_argument("--train", required=True, help="Path to train JSONL.")
    parser.add_argument("--val", required=True, help="Path to validation JSONL.")
    parser.add_argument("--out", required=True, help="Directory to store adapters.")
    parser.add_argument(
        "--model",
        default=os.getenv("SLM_MODEL_PATH"),
        help="Base SLM path (defaults to SLM_MODEL_PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading/training; only validate datasets and emit metadata.",
    )
    args = parser.parse_args()

    if not args.model and not args.dry_run:
        raise EnvironmentError("Provide --model or set SLM_MODEL_PATH.")

    os.makedirs(args.out, exist_ok=True)
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    max_seq_length = 2048
    missing_env = [key for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not os.getenv(key)]
    use_null_trace = args.dry_run and missing_env
    trace_ctx = (
        langfuse_trace(
            name=f"train-unsloth-{args.track}",
            metadata={
                "track": args.track,
                "train_path": args.train,
                "val_path": args.val,
                "output_dir": args.out,
                "dry_run": args.dry_run,
            },
        )
        if not use_null_trace
        else nullcontext()
    )
    with trace_ctx as trace:
        if args.dry_run:
            stats = {
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "avg_train_chars": sum(len(row["prompt"]) for row in train_rows) / max(len(train_rows), 1),
                "avg_val_chars": sum(len(row["prompt"]) for row in val_rows) / max(len(val_rows), 1),
            }
            summary_path = os.path.join(args.out, f"dry_run_{args.track}.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)
            if trace:
                trace.update(metadata={"dry_run": True, "dataset_stats": stats, "summary_path": summary_path})
            print(json.dumps({"dry_run": True, **stats}, indent=2))
            return

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            target_modules="all-linear",
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        formatted_train = format_dataset(train_ds, tokenizer, args.track)
        formatted_val = format_dataset(val_ds, tokenizer, args.track)

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            output_dir=args.out,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_train,
            eval_dataset=formatted_val,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
            args=training_args,
        )
        train_result = trainer.train()
        metrics = {
            "train_loss": float(train_result.training_loss or 0.0),
        }
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        if trace:
            trace.update(
                metadata={
                    "hyperparams": {
                        "r": 64,
                        "alpha": 16,
                        "dropout": 0.1,
                        "lr": 2e-4,
                        "epochs": 1,
                        "seq_len": max_seq_length,
                    },
                    "metrics": metrics,
                }
            )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
