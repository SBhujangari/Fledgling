# FUNCTION_CALLING_FINETUNE Approach - Docker Setup

This directory contains the Docker configuration for the **paper-based approach** following the FUNCTION_CALLING_FINETUNE.md methodology.

## Architecture

- **Base Image:** `unsloth/unsloth:latest` (official Unsloth Docker image)
- **Models:** Supports both Phi-4-mini and Llama-3.1-8B-Instruct
- **Dataset:** Hermes Function Calling v1
- **Training:** QLoRA with standard paper hyperparameters

## GPU Allocation

The docker-compose configuration runs **two parallel experiments**:

- **Container 1 (paper-approach-phi):** Phi-4-mini on GPUs 0,1
- **Container 2 (paper-approach-llama):** Llama-3.1-8B-Instruct on GPUs 2,3

## Quick Start

### 1. Build the Container

```bash
cd /home/gabriel/Desktop/AI_ATL25/docker/unsloth-paper
docker-compose build
```

### 2. Prepare Hermes Dataset

First, download and prepare the Hermes Function Calling v1 dataset (see main README).

### 3. Run Both Models in Parallel

```bash
# Start both containers
docker-compose up -d

# Training on Phi-4-mini (GPUs 0,1)
docker-compose exec unsloth-paper-phi bash
cd paper_approach/scripts
python train_function_call.py \
  --model-id microsoft/phi-4-mini \
  --dataset-path ../datasets/hermes_train.jsonl \
  --val-dataset-path ../datasets/hermes_val.jsonl \
  --output-dir ../adapters/phi_hermes_lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --epochs 3

# Training on Llama-3.1-8B (GPUs 2,3) - in separate terminal
docker-compose exec unsloth-paper-llama bash
cd paper_approach/scripts
python train_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path ../datasets/hermes_train.jsonl \
  --val-dataset-path ../datasets/hermes_val.jsonl \
  --output-dir ../adapters/llama_hermes_lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --epochs 3
```

### 4. Evaluate

```bash
# Evaluate Phi-4-mini
docker-compose exec unsloth-paper-phi bash -c "cd paper_approach/scripts && python eval_function_call.py \
  --model-id microsoft/phi-4-mini \
  --adapter-path ../adapters/phi_hermes_lora \
  --dataset-path ../datasets/hermes_test.jsonl \
  --output-path ../eval_results/phi_hermes_results.json"

# Evaluate Llama-3.1-8B
docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && python eval_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path ../adapters/llama_hermes_lora \
  --dataset-path ../datasets/hermes_test.jsonl \
  --output-path ../eval_results/llama_hermes_results.json"
```

## Hyperparameters (Paper Defaults)

Following FUNCTION_CALLING_FINETUNE.md:

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Optimizer | AdamW 8-bit |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Max seq length | 2048 |
| Precision | BF16 |

## Output Structure

```
paper_approach/
├── datasets/
│   ├── hermes_train.jsonl
│   ├── hermes_val.jsonl
│   └── hermes_test.jsonl
│
├── adapters/
│   ├── phi_hermes_lora/
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   └── metadata.json
│   └── llama_hermes_lora/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── metadata.json
│
├── eval_results/
│   ├── phi_hermes_results.json
│   ├── phi_hermes_results_details.jsonl
│   ├── llama_hermes_results.json
│   └── llama_hermes_results_details.jsonl
│
└── logs/
    ├── phi_training.log
    └── llama_training.log
```

## Evaluation Metrics

The paper approach tracks:

- **valid_json_rate:** Percentage of outputs that parse as valid JSON
- **name_match_rate:** Tool name accuracy
- **args_exact_match_rate:** Perfect argument match (canonical JSON)
- **args_field_precision:** Per-field precision
- **args_field_recall:** Per-field recall
- **args_field_f1:** Harmonic mean of precision and recall

## Comparison to Fledgling

| Aspect | Fledgling (Current) | Paper Approach |
|--------|---------------------|----------------|
| Base Models | Phi-4-mini | Phi-4-mini + Llama-3.1-8B |
| Dataset | xLAM (60k examples) | Hermes Function Calling v1 |
| Tracks | 2 (structured + toolcall) | 1 (unified function calling) |
| LoRA rank | 64 | 16 |
| Training approach | Azure comparison → decision gate | Direct fine-tuning |
| Evaluation | Azure baseline + SLM | SLM only |
| Docker | Custom CUDA image | Official Unsloth image |

## Troubleshooting

### Container Won't Start

Check GPU allocation:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Hugging Face Access Denied

Ensure `HUGGING_FACE_HUB_TOKEN` is set in `.env` and you have access to:
- `meta-llama/Llama-3.1-8B-Instruct` (gated - requires Meta approval)
- Hermes Function Calling dataset (check access permissions)

### Out of Memory

- Reduce `--batch-size` to 2 or 1
- Increase `--grad-accum` to maintain effective batch size
- Reduce `--lora-r` to 8

## Next Steps

After training both models, use the comparison framework (to be implemented) to compare:
1. Fledgling vs Paper approach
2. Phi-4-mini vs Llama-3.1-8B
3. xLAM vs Hermes dataset
