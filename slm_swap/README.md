# SLM Swap v0

Minimal, evaluation-first workflow to compare a local SLM against an Azure-hosted LLM across Structured JSON and Tool-calling tracks. Langfuse tracing is required for every evaluation and training run.

## Prerequisites
- Ubuntu + Python 3.11+
- Four NVIDIA RTX 3090 GPUs (fine-tuning assumes all four are available)
- Python deps (install once from the repo root via `pip install -r requirements.txt`):
  - `torch`, `transformers`, `accelerate`, `bitsandbytes`, `deepspeed`
  - `langfuse`, `openai`
  - `unsloth`, `datasets`, `trl`
- Local SLM checkpoint downloaded into `slm_swap/models/` (default `phi-4-mini`)
- Langfuse workspace with API keys (set `LANGFUSE_DISABLED=1` only for local dry-runs)

## Environment Variables
Export before running anything:
```
export AZURE_ENDPOINT="https://<endpoint>/openai/deployments/<deployment>"
export AZURE_DEPLOYMENT="chat"
export AZURE_API_KEY="..."
export AZURE_API_VERSION="2024-02-15-preview"
export SLM_MODEL_PATH="slm_swap/models/phi-4-mini"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
# Optional local bypass
# export LANGFUSE_DISABLED=1
```

## Local SLM Download (Phi-4-mini)
```
huggingface-cli download microsoft/phi-4-mini --local-dir slm_swap/models/phi-4-mini
```
The `slm_client` loads the checkpoint with `device_map="auto"` and 4-bit quantization so inference is automatically sharded across the four GPUs.

## Dataset Preparation
The tiny splits must be derived from the gated `Salesforce/xlam-function-calling-60k` dataset.

1. Login and download the dataset (requires manual access approval from Salesforce):
   ```bash
   huggingface-cli login  # only once
   huggingface-cli download Salesforce/xlam-function-calling-60k \
     --repo-type dataset \
     --local-dir raw_xlam
   ```
2. Generate the structured/tool-call JSONLs (defaults to 32/8/8 examples per split):
   ```bash
   python prepare_data.py --dataset-path raw_xlam/xlam_function_calling_60k.json
   ```

This produces the required layout:
```
slm_swap/02_dataset/
  structured/{train,val,test}.jsonl
  toolcall/{train,val,test}.jsonl
```
Each JSONL row is `{ "prompt": ..., "completion": ... }`. Structured completions are pure JSON objects describing the chosen tool/arguments, while tool-calling completions are `<tool_call name="...">{...}</tool_call>` wrappers with observations removed.

## Baseline Evaluations
1. **Structured track**
   - Azure reference:
     ```bash
     python eval.py --track structured --model-kind azure --split test --out 05_eval/structured_azure_test.json
     ```
   - Local SLM:
     ```bash
     python eval.py --track structured --model-kind slm --split test --out 05_eval/structured_slm_test.json
     ```
2. **Tool-calling track**
   - Azure reference:
     ```bash
     python eval.py --track toolcall --model-kind azure --split test --out 05_eval/toolcall_azure_test.json
     ```
   - Local SLM:
     ```bash
     python eval.py --track toolcall --model-kind slm --split test --out 05_eval/toolcall_slm_test.json
     ```
Each run emits metrics JSON and logs a Langfuse trace that captures dataset path, model kind, metrics path, start, and end.

## Comparison / Decision
Decide whether to fine-tune using `compare.py` with a positive delta threshold per track:
```bash
python compare.py --track structured --azure 05_eval/structured_azure_test.json --slm 05_eval/structured_slm_test.json --delta 0.01
python compare.py --track toolcall --azure 05_eval/toolcall_azure_test.json --slm 05_eval/toolcall_slm_test.json --delta 0.01
```
The script prints `FINE_TUNE=1` if the SLM trails Azure on any core metric by more than `delta`, logs a Langfuse event, and exits with status 10 to signal downstream automation.

## Conditional Fine-Tune (Unsloth QLoRA on 4×3090)
Run *only* when the compare step requests it. Launch with `accelerate` (preferred) or `torchrun` so all GPUs participate; the trainer automatically wires `accelerate_config/deepspeed_zero2.json` for ZeRO-2 sharding.
```bash
# Accelerate example
accelerate launch --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured

# Torchrun alternative
torchrun --nproc_per_node=4 train_unsloth.py --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall
```
Fixed hyperparameters: 4-bit load, LoRA `r=64`, `alpha=16`, `dropout=0.1`, `lr=2e-4`, `epochs=1`, `seq_len≈2048`. Training traces capture dataset paths, hyperparameters, adapter path, and multi-GPU strategy.

## Re-evaluation After Fine-Tune
Reuse `eval.py` with the same commands, pointing `SLM_MODEL_PATH` to the base checkpoint and loading the generated adapter (see script flag). Store metrics under `05_eval/` (e.g., `structured_slm_ft_test.json`) and trace every run.

## Optional Sanity Inference
`infer.py` can run a one-off request per track against either client to confirm wiring before full evals.

## Minimal Mastra Integration Note
Mastra can call the local SLM by reusing the `slm_client` helper: construct the same `[system, user]` messages used for evaluation, load adapters if present, and surface the raw text response—no routing, fallbacks, or judges in v0.

## Langfuse Requirement
No eval or training command should run without `LANGFUSE_*` configured (unless explicitly bypassed via `LANGFUSE_DISABLED=1` during local smoke tests). Expect traces for:
- Every `eval.py` invocation (structured/toolcall × Azure/SLM × splits)
- Every `train_unsloth.py` invocation
- Every `compare.py` decision event
