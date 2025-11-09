#!/bin/bash
# Sequential single-GPU training: Train structured then toolcall
# Simple approach to get baseline working

set -e

echo "========================================="
echo "Starting sequential single-GPU training"
echo "========================================="

# Train structured model on GPU 0
echo "Training structured model on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_unsloth.py \
  --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured

echo ""
echo "Structured training complete!"
echo ""

# Train toolcall model on GPU 0
echo "Training toolcall model on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_unsloth.py \
  --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall

echo ""
echo "========================================="
echo "✓ Both models trained successfully!"
echo "  Structured adapter: 04_ft/adapter_structured"
echo "  Toolcall adapter:   04_ft/adapter_toolcall"
echo "========================================="
