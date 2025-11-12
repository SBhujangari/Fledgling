#!/bin/bash
# Entrypoint script for Fledgling Docker container

set -e

echo "============================================="
echo "Fledgling Pipeline Container Starting"
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
else
    echo "WARNING: /workspace/.env not found. API keys may not be configured."
fi

# Check critical environment variables
if [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
    echo "WARNING: LANGFUSE_PUBLIC_KEY not set. Langfuse tracing may fail."
fi

if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "WARNING: AZURE_OPENAI_API_KEY not set. Azure baseline may fail."
fi

echo ""
echo "Environment configured. Ready to run Fledgling pipeline."
echo "============================================="
echo ""

# Execute the command passed to docker run
exec "$@"
