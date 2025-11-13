# SLM Swap v0

Minimal, evaluation-first workflow to compare a local SLM against an Azure-hosted LLM across Structured JSON and Tool-calling tracks. Langfuse tracing is required for every evaluation and training run.

## Prerequisites
- Ubuntu + Python 3.11+
- GPU with enough VRAM for 4-bit QLoRA (tested with 24 GB)
- Python deps (install once from the repo root via `pip install -r requirements.txt`):
  - `torch`, `transformers`, `accelerate`, `bitsandbytes`
  - `langfuse`, `openai`
  - `unsloth`, `datasets`, `trl`
- Local SLM checkpoint downloaded into `slm_swap/models/` (default `qwen2.5-7b-instruct`)
- Langfuse workspace with API keys

## Environment Variables
Export before running anything:
```
export AZURE_ENDPOINT="https://<endpoint>/openai/deployments/<deployment>"
export AZURE_API_KEY="..."
export AZURE_API_VERSION="2024-02-15-preview"
export SLM_MODEL_PATH="slm_swap/models/qwen2.5-7b-instruct"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

## Dataset Preparation

### Option A — Harvest live traces via Langfuse (recommended)
Automatically convert tool-using agent traces into training/eval splits:

```bash
cd slm_swap
python langfuse_dataset.py \
  --agent-id demo-agent \
  --limit 400 \
  --output-root 02_dataset \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

The script:
- Pulls traces for the given `agent_id` directly from Langfuse
- Extracts user queries + tool calls (arguments + tool names)
- Emits `structured/` JSON targets and `<tool_call>` wrappers that match the rest of the pipeline
- Logs an ingestion trace back to Langfuse for provenance

### Option B — Bootstrap from xLAM (offline smoke tests)
If you don't have live traces yet, derive tiny splits from the gated `Salesforce/xlam-function-calling-60k` dataset:

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

Both paths produce:
```
slm_swap/02_dataset/
  structured/{train,val,test}.jsonl
  toolcall/{train,val,test}.jsonl
```
Each JSONL row is `{ "prompt": ..., "completion": ... }`. Structured completions are pure JSON objects describing the chosen tool/arguments, while tool-calling completions are `<tool_call name="...">{...}</tool_call>` wrappers with observations removed.

### Option C — Built-in dummy agent (fully offline)
When you just want to smoke-test the Langfuse → dataset loop without credentials:
```bash
python dummy_agent_workflow.py \
  --requests ../../datasets/dummy_agent_requests.jsonl \
  --out-json ../../storage/dummy_langfuse_traces.jsonl \
  --agent-id dummy-agent

python langfuse_dataset.py \
  --trace-json ../../storage/dummy_langfuse_traces.jsonl \
  --agent-id dummy-agent \
  --output-root 02_dataset/dummy_langfuse
```
Add `--emit-to-langfuse` to the first command if you want the traces to land in a real workspace as well.

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

## Conditional Fine-Tune (Unsloth QLoRA)
Run *only* when the compare step requests it.
```bash
python train_unsloth.py --track structured --train 02_dataset/structured/train.jsonl --val 02_dataset/structured/val.jsonl --out 04_ft/adapter_structured
python train_unsloth.py --track toolcall --train 02_dataset/toolcall/train.jsonl --val 02_dataset/toolcall/val.jsonl --out 04_ft/adapter_toolcall
```
Fixed hyperparameters: 4-bit load, LoRA `r=64`, `alpha=16`, `dropout=0.1`, `lr=2e-4`, `epochs=1`, `seq_len≈2048`. Training traces capture dataset paths, hyperparameters, and adapter save location.

Need a fast wiring test with no GPU? Append `--dry-run` to the commands above. The script will validate the datasets, emit stats, and skip model loading while still logging a Langfuse trace.

## Re-evaluation After Fine-Tune
Reuse `eval.py` with the same commands, pointing `SLM_MODEL_PATH` to the base checkpoint and loading the generated adapter (see script flag). Store metrics under `05_eval/` (e.g., `structured_slm_ft_test.json`) and trace every run.

## Optional Sanity Inference
`infer.py` can run a one-off request per track against either client to confirm wiring before full evals.

## Minimal Mastra Integration Note
Mastra can call the local SLM by reusing the `slm_client` helper: construct the same `[system, user]` messages used for evaluation, load adapters if present, and surface the raw text response—no routing, fallbacks, or judges in v0.

## Langfuse Requirement
No eval or training command should run without `LANGFUSE_*` configured. Expect traces for:
- Every `eval.py` invocation (structured/toolcall × Azure/SLM × splits)
- Every `train_unsloth.py` invocation
- Every `compare.py` decision event

## Dummy Pipeline Runner
For CI-style wiring tests, run everything (dummy traces → dataset → dry-run fine-tunes) via:
```bash
python run_dummy_pipeline.py --clean --count 12
```
This orchestrates `dummy_agent_workflow.py`, `langfuse_dataset.py --trace-json ...`, and `train_unsloth.py --dry-run` for both tracks, leaving artifacts under `02_dataset/dummy_langfuse/` and `04_ft/dummy_langfuse/`.
It also emits `slm_swap/logs/finetune_progress.json` so `/api/training/status` (and the frontend tuning dashboard) have live data without touching real GPUs.
