"""Conditional Unsloth QLoRA fine-tuning for the local SLM."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from numbers import Number
from typing import Dict, List

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from eval import StructuredMetricTracker, ToolcallMetricTracker
from langfuse_helpers import langfuse_trace
from model_version import generate_model_id
from prompts import get_system_prompt
from slm_client import DEFAULT_SLM_MODEL_PATH
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _unwrap_model(model):
    return getattr(model, "module", model)


def _sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, Number):
            clean[key] = float(value)
        else:
            try:
                clean[key] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return clean


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    track: str,
    max_new_tokens: int = 256,
) -> str:
    system_prompt = get_system_prompt(track)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def compute_schema_metrics(
    model,
    tokenizer,
    val_rows: List[Dict[str, str]],
    track: str,
) -> Dict[str, float]:
    if not val_rows:
        return {}

    model_inference = FastLanguageModel.for_inference(model)
    was_training = model_inference.training
    model_inference.eval()
    tracker = (
        StructuredMetricTracker() if track == "structured" else ToolcallMetricTracker()
    )
    for row in val_rows:
        prediction = generate_completion(
            model_inference,
            tokenizer,
            prompt=row["prompt"],
            track=track,
        )
        tracker.update(prediction, row["completion"])
    if was_training:
        model_inference.train()
    return tracker.summary()


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
        default=os.getenv("SLM_MODEL_PATH", DEFAULT_SLM_MODEL_PATH),
        help="Base SLM path (defaults to SLM_MODEL_PATH or phi-4-mini).",
    )
    parser.add_argument(
        "--deepspeed-config",
        default=os.path.join(
            os.path.dirname(__file__), "accelerate_config", "deepspeed_zero2.json"
        ),
        help="Path to DeepSpeed ZeRO config (set empty to disable).",
    )
    args = parser.parse_args()

    if not args.model:
        raise EnvironmentError("Provide --model or set SLM_MODEL_PATH.")

    os.makedirs(args.out, exist_ok=True)
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    max_seq_length = 2048
    dataset_info = {
        "track": args.track,
        "train_path": args.train,
        "val_path": args.val,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    hyperparams = {
        "r": 64,
        "alpha": 16,
        "dropout": 0.1,
        "lr": 2e-4,
        "epochs": 3,
        "seq_len": max_seq_length,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "bf16": True,
        "deepspeed_config": args.deepspeed_config or None,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
    }
    version_timestamp = (
        os.environ.get("MODEL_VERSION_TIMESTAMP") or datetime.utcnow().isoformat()
    )
    model_version_id = generate_model_id(
        hyperparams, dataset_info, timestamp=version_timestamp
    )
    is_main = _is_main_process()
    with langfuse_trace(
        name=f"train-unsloth-{args.track}",
        metadata={
            "track": args.track,
            "train_path": args.train,
            "val_path": args.val,
            "output_dir": args.out,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "model_version_id": model_version_id,
        },
    ) as trace:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect bf16
            load_in_4bit=False,
            load_in_8bit=False,  # Disable quantization for multi-GPU training
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
            per_device_train_batch_size=hyperparams["batch_size"],
            per_device_eval_batch_size=hyperparams["batch_size"],
            gradient_accumulation_steps=hyperparams["grad_accum_steps"],
            learning_rate=hyperparams["lr"],
            num_train_epochs=hyperparams["epochs"],
            bf16=hyperparams["bf16"],
            logging_steps=1,
            logging_strategy="steps",
            logging_first_step=True,
            eval_strategy="epoch",
            lr_scheduler_type=hyperparams["lr_scheduler"],
            warmup_ratio=hyperparams["warmup_ratio"],
            save_strategy="no",
            report_to="none",
            output_dir=args.out,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            deepspeed=args.deepspeed_config or None,
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
        hf_eval_metrics = _sanitize_metrics(trainer.evaluate())
        eval_loss = float(hf_eval_metrics.get("eval_loss", 0.0))
        schema_metrics: Dict[str, float] = {}
        if is_main:
            eval_model = _unwrap_model(trainer.model)
            schema_metrics = compute_schema_metrics(
                eval_model,
                tokenizer,
                val_rows,
                args.track,
            )
            hf_eval_metrics.update(
                {f"schema_{k}": v for k, v in (schema_metrics or {}).items()}
            )
        version_metadata = {
            "model_version_id": model_version_id,
            "timestamp": version_timestamp,
            "hyperparams": hyperparams,
            "dataset": dataset_info,
            "metrics": {
                "train": metrics,
                "validation": {
                    "eval_loss": eval_loss,
                    "hf_eval": hf_eval_metrics,
                    "schema": schema_metrics,
                },
            },
        }
        if is_main:
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            with open(os.path.join(args.out, "VERSION.txt"), "w", encoding="utf-8") as f:
                f.write(model_version_id + "\n")
            with open(
                os.path.join(args.out, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(version_metadata, f, indent=2)
            trace.update(
                metadata={
                    "hyperparams": hyperparams,
                    "metrics": {
                        "train": metrics,
                        "validation": {
                            "eval_loss": eval_loss,
                            "hf_eval": hf_eval_metrics,
                            "schema": schema_metrics,
                        },
                    },
                    "model_version_id": model_version_id,
                    "timestamp": version_timestamp,
                }
            )
            print(
                json.dumps(
                    {
                        "model_version_id": model_version_id,
                        "train_metrics": metrics,
                        "validation_metrics": {
                            "eval_loss": eval_loss,
                            "schema": schema_metrics,
                        },
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
