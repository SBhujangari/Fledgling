#!/bin/bash
# Sequential training: Train structured and toolcall models one after another using all 4 GPUs
# This may be faster per-model but takes 2x wallclock time vs parallel training

set -e

# Check if we're in the slm_swap directory
if [ ! -f "train_unsloth.py" ]; then
    echo "Error: Must run from slm_swap/ directory"
    exit 1
fi

mkdir -p logs

echo "========================================="
echo "Starting sequential training (4 GPUs each)"
echo "========================================="
echo ""

# Train structured model on all 4 GPUs
echo "[1/2] Training structured model on GPUs 0,1,2,3..."
accelerate launch \
  --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py \
  --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured \
  2>&1 | tee logs/train_structured_4gpu.log

STRUCTURED_EXIT=${PIPESTATUS[0]}
if [ $STRUCTURED_EXIT -ne 0 ]; then
    echo "✗ Structured training failed (exit code: $STRUCTURED_EXIT)"
    exit $STRUCTURED_EXIT
fi

echo "✓ Structured training complete"
echo ""

# Train toolcall model on all 4 GPUs
echo "[2/2] Training toolcall model on GPUs 0,1,2,3..."
accelerate launch \
  --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py \
  --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall \
  2>&1 | tee logs/train_toolcall_4gpu.log

TOOLCALL_EXIT=${PIPESTATUS[0]}
if [ $TOOLCALL_EXIT -ne 0 ]; then
    echo "✗ Toolcall training failed (exit code: $TOOLCALL_EXIT)"
    exit $TOOLCALL_EXIT
fi

echo ""
echo "========================================="
echo "✓ Both training jobs completed successfully"
echo "  Structured adapter: 04_ft/adapter_structured"
echo "  Toolcall adapter:   04_ft/adapter_toolcall"
echo "========================================="
