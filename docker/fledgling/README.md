# Fledgling Pipeline - Docker Setup

This directory contains the Docker configuration for the **current Fledgling pipeline approach** (Phi-4-mini with Azure comparison).

## Quick Start

### 1. Build the Container

```bash
cd /home/gabriel/Desktop/AI_ATL25/docker/fledgling
docker-compose build
```

### 2. Run the Container

```bash
docker-compose up -d
docker-compose exec fledgling bash
```

### 3. Run Evaluation Pipeline

Inside the container:

```bash
cd slm_swap

# Run baseline evaluation
bash run_all_evals.sh

# Run comparison
bash run_comparison.sh

# If fine-tuning is triggered
bash train_parallel.sh  # or train_sequential.sh
```

## Architecture

- **Base Image:** NVIDIA CUDA 12.1.0 + cuDNN 8 on Ubuntu 22.04
- **Python:** 3.11
- **GPU Support:** All available NVIDIA GPUs (4x RTX 3090)
- **Shared Memory:** 16GB (for multi-GPU training)

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `../../slm_swap/models` | `/workspace/slm_swap/models` | Pre-downloaded model weights |
| `../../slm_swap/02_dataset` | `/workspace/slm_swap/02_dataset` | Training/eval datasets |
| `../../slm_swap/04_ft` | `/workspace/slm_swap/04_ft` | Fine-tuned adapters |
| `../../slm_swap/05_eval` | `/workspace/slm_swap/05_eval` | Evaluation results |
| `../../slm_swap/logs` | `/workspace/slm_swap/logs` | Training logs |

## Environment Variables

Create a `.env` file in the project root with:

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Hugging Face (optional)
HUGGING_FACE_HUB_TOKEN=hf_...
```

## Useful Commands

### Check GPU Status

```bash
docker-compose exec fledgling nvidia-smi
```

### Run Live Comparison

```bash
docker-compose exec fledgling bash -c "cd slm_swap && python live_compare.py --track structured --models azure slm"
```

### Train with Custom Config

```bash
docker-compose exec fledgling bash -c "cd slm_swap && python train_unsloth.py --track structured --train-path 02_dataset/structured/train.jsonl --val-path 02_dataset/structured/val.jsonl --output-dir 04_ft/adapter_structured"
```

### Stop Container

```bash
docker-compose down
```

## Troubleshooting

### Out of Memory

- Reduce batch size in `train_unsloth.py` (default: 1)
- Increase gradient accumulation steps
- Use `train_sequential.sh` instead of `train_parallel.sh`

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If failed, install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Langfuse Connection Issues

Set `LANGFUSE_DISABLED=1` in `.env` to disable tracing (for testing only).

## Performance Notes

- **4x RTX 3090 (12GB each):** Can train Phi-4-mini (3.8B) with QLoRA (rank 64)
- **Training time:** ~45 minutes for 300 examples × 3 epochs (structured track)
- **Inference batch size:** 8 for SLM evaluation (leverages all GPUs)
