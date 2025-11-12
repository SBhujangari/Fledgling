#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_PATH="${LOG_DIR}/train_llama_cep_pure.log"
PID_PATH="${LOG_DIR}/train_llama_cep_pure.pid"

MODEL_ID="${MODEL_ID:-unsloth/llama-3.1-8b-instruct-bnb-4bit}"
TRAIN_JSON="${TRAIN_JSON:-${REPO_ROOT}/paper_approach/datasets/hermes_train.jsonl}"
VAL_JSON="${VAL_JSON:-${REPO_ROOT}/paper_approach/datasets/hermes_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/04_ft/adapter_llama_cep}"
CEP_TYPE="${CEP_TYPE:-universal}"

mkdir -p "${LOG_DIR}"

echo "== GPU health check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA drivers before launching training."
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed. GPUs are not visible (likely 'GPU has fallen off the bus')."
  echo "Check 'dmesg | tail' and power-cycle / reload the NVIDIA driver, then rerun this script."
  exit 1
fi

if [ ! -f "${TRAIN_JSON}" ]; then
  echo "ERROR: Training dataset missing at ${TRAIN_JSON}"
  exit 1
fi

VAL_ARGS=()
if [ -n "${VAL_JSON}" ] && [ -f "${VAL_JSON}" ]; then
  VAL_ARGS=(--val "${VAL_JSON}")
else
  echo "⚠️  Validation set not found at ${VAL_JSON}; continuing without eval."
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Launching training with GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Memory optimization: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
nohup python3 "${SCRIPT_DIR}/train_llama_cep_pure.py" \
  --model-id "${MODEL_ID}" \
  --train "${TRAIN_JSON}" \
  --output "${OUTPUT_DIR}" \
  --cep-type "${CEP_TYPE}" \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536 \
  "${VAL_ARGS[@]}" \
  > "${LOG_PATH}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_PATH}"

echo ""
echo "Training started (PID ${PID})."
echo "Log: ${LOG_PATH}"
echo "Tail logs with: tail -f ${LOG_PATH}"
