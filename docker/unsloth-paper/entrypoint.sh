#!/bin/bash
# Entrypoint script for FUNCTION_CALLING_FINETUNE Paper Approach

set -e

echo "============================================="
echo "FUNCTION_CALLING_FINETUNE Container Starting"
echo "Paper-based Approach (Hermes + Llama/Phi)"
echo "============================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Set deterministic environment variables
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Load environment variables from .env if it exists
if [ -f "/workspace/.env" ]; then
    echo "Loading environment variables from /workspace/.env"
    set -a
    source /workspace/.env
    set +a
fi

# Check Hugging Face token for model/dataset downloads
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Hugging Face token detected. Logging in..."
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
else
    echo "WARNING: HUGGING_FACE_HUB_TOKEN not set. Model/dataset downloads may fail for gated repos."
fi

echo ""
echo "Environment configured. Ready for paper-based training."
echo "============================================="
echo ""

# Execute the command passed to docker run
exec "$@"
