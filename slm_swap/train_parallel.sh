#!/bin/bash
# Parallel training: Train structured and toolcall models simultaneously using 2 GPUs each
# This script launches two training jobs in parallel, each using 2 GPUs with DeepSpeed ZeRO-2

set -e

# Check if we're in the slm_swap directory
if [ ! -f "train_unsloth.py" ]; then
    echo "Error: Must run from slm_swap/ directory"
    exit 1
fi

timestamp() {
  python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
}

# Ensure accelerate config exists
if [ ! -f "accelerate_config/phi4_2gpu.yaml" ]; then
    echo "Error: Missing accelerate_config/phi4_2gpu.yaml"
    echo "Creating 2-GPU config..."
    cat > accelerate_config/phi4_2gpu.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
machine_rank: 0
main_process_ip: null
main_process_port: null
mixed_precision: bf16
use_cpu: false
downcast_bf16: no
deepspeed_config:
  deepspeed_config_file: accelerate_config/deepspeed_zero2.json
  zero3_init_flag: false
EOF
fi

STRUCTURED_TS=$(timestamp)
TOOLCALL_TS=$(timestamp)

echo "========================================="
echo "Starting parallel training (2+2 GPUs)"
echo "========================================="
echo "Structured: GPUs 0,1"
echo "Toolcall:   GPUs 2,3"
echo "Structured timestamp: $STRUCTURED_TS"
echo "Toolcall timestamp:   $TOOLCALL_TS"
echo ""

# Train structured model on GPUs 0,1 (port 29500)
MODEL_VERSION_TIMESTAMP=$STRUCTURED_TS CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_config/phi4_2gpu.yaml \
  --main_process_port 29500 \
  train_unsloth.py \
  --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured \
  > logs/train_structured_2gpu.log 2>&1 &

STRUCTURED_PID=$!
echo "Structured training started (PID: $STRUCTURED_PID)"

# Train toolcall model on GPUs 2,3 (port 29501 to avoid conflict)
MODEL_VERSION_TIMESTAMP=$TOOLCALL_TS CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file accelerate_config/phi4_2gpu.yaml \
  --main_process_port 29501 \
  train_unsloth.py \
  --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall \
  > logs/train_toolcall_2gpu.log 2>&1 &

TOOLCALL_PID=$!
echo "Toolcall training started (PID: $TOOLCALL_PID)"

echo ""
echo "Monitor logs:"
echo "  tail -f logs/train_structured_2gpu.log"
echo "  tail -f logs/train_toolcall_2gpu.log"
echo ""
echo "Waiting for both training jobs to complete..."

# Wait for both processes
wait $STRUCTURED_PID
STRUCTURED_EXIT=$?

wait $TOOLCALL_PID
TOOLCALL_EXIT=$?

echo ""
echo "========================================="
if [ $STRUCTURED_EXIT -eq 0 ] && [ $TOOLCALL_EXIT -eq 0 ]; then
    echo "✓ Both training jobs completed successfully"
    echo "  Structured adapter: 04_ft/adapter_structured"
    echo "  Toolcall adapter:   04_ft/adapter_toolcall"
    exit 0
else
    echo "✗ Training failed:"
    [ $STRUCTURED_EXIT -ne 0 ] && echo "  Structured exit code: $STRUCTURED_EXIT"
    [ $TOOLCALL_EXIT -ne 0 ] && echo "  Toolcall exit code: $TOOLCALL_EXIT"
    exit 1
fi
