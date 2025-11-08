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
export AZURE_API_VERSION="2024-05-01-preview"
export SLM_MODEL_PATH="slm_swap/models/phi-4-mini"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
# Optional local bypass
# export LANGFUSE_DISABLED=1
```
> **Azure endpoint tip:** the default client expects the canonical `https://<resource>.openai.azure.com/openai/deployments/<deployment>` shape. If your resource exposes the preview `models/chat/completions` path instead, make sure it actually exists and that `AZURE_DEPLOYMENT` matches the configured deployment name; otherwise Azure will return `DeploymentNotFound`.

## Local SLM Download (Phi-4-mini)
```
huggingface-cli download microsoft/phi-4-mini --local-dir slm_swap/models/phi-4-mini
```
The `slm_client` loads the checkpoint with `device_map="auto"` and 8-bit quantization so inference is automatically sharded across the four GPUs.

## Determinism and Reproducibility
All evaluation and training runs enforce strict determinism to ensure reproducible results:
- **Random seeds**: Fixed at 42 for Python `random`, NumPy, PyTorch (CPU/CUDA), and transformers
- **SLM inference**: Model in `.eval()` mode (dropout disabled), greedy decoding (`do_sample=False`), `temperature=0`, `top_p=1.0`, `top_k=0`, CUDA deterministic operations enabled
- **Azure LLM**: `temperature=0`, `seed=42` passed to API for reproducible sampling
- **Environment**: `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms=True`

This guarantees bit-for-bit identical outputs across runs given the same inputs, hardware, and software versions.

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

## 50-Case Smoke Test (Structured + Toolcall)
If you just need a small batch of test cases and want to watch both tracks run with the live dashboard, use the helper script (run from `slm_swap/`):

```
cd slm_swap
export HF_HOME="$PWD/.hf_modules"
export SLM_MODEL_PATH="$PWD/models/phi-4-mini"
# export AZURE_ENDPOINT/KEY/VERSION[/DEPLOYMENT] first if you plan to include the Azure teacher
python run_first_tests.py --dataset-path raw_xlam/xlam_function_calling_60k.json --models azure slm --save
```

This does three things:
1. Resamples 100 fresh rows from xLAM so Structured and Toolcall each get their **own** 50-prompt test split (`02_dataset/structured/test.jsonl` and `02_dataset/toolcall/test.jsonl`).
2. Runs `live_compare.py` for the Structured track so you can watch Azure vs SLM progress update after every case.
3. Repeats the live view for the Toolcall track.

Use `--models slm` if you only want the local model, `--adapter` to load a fine-tuned LoRA, and `--dataset-path` if your xLAM dump lives elsewhere. Metrics are saved under `05_eval/` when `--save` is set, so you can still feed them to `compare.py` afterward.

> **Tip:** When you're already inside `slm_swap/`, set `SLM_MODEL_PATH` to `models/phi-4-mini` (or its absolute path). Prefixing it with `slm_swap/…` causes the loader to look for `/slm_swap/slm_swap/models/...`, which doesn't exist.

## Baseline Evaluations

**SLM evaluation uses batched inference** (default `--batch-size 8`) to process multiple examples in parallel across 4 GPUs for maximum throughput. The model is tensor-parallelized (layers distributed across GPUs), and batching enables concurrent processing of examples.

1. **Structured track**
   - Azure reference:
     ```bash
     python eval.py --track structured --model-kind azure --split test --out 05_eval/structured_azure_test.json
     ```
   - Local SLM (batched):
     ```bash
     python eval.py --track structured --model-kind slm --split test --out 05_eval/structured_slm_test.json --batch-size 8
     ```
2. **Tool-calling track**
   - Azure reference:
     ```bash
     python eval.py --track toolcall --model-kind azure --split test --out 05_eval/toolcall_azure_test.json
     ```
   - Local SLM (batched):
     ```bash
     python eval.py --track toolcall --model-kind slm --split test --out 05_eval/toolcall_slm_test.json --batch-size 8
     ```

Adjust `--batch-size` based on available VRAM. With 4×3090s (96GB total), batch sizes of 8-16 typically work well. Use `--batch-size 1` for sequential processing.

Each run emits metrics JSON and logs a Langfuse trace that captures dataset path, model kind, metrics path, start, and end.

## Live Evaluation View (Terminal)
Use `live_compare.py` to watch Azure and the local SLM progress on the same dataset in real time. The script streams per-example metric snapshots, shows the current Azure–SLM gaps, and can optionally persist the final JSON files for use with `compare.py`.

```
python live_compare.py --track structured --split test --models azure slm --save
```

- Omit `--save` to run the viewer without overwriting anything under `05_eval/`.
- Pass `--models slm` (or `azure`) to focus on a single client.
- Add `--adapter 04_ft/adapter_structured` when viewing a fine-tuned SLM.

This CLI output is the seed for the future UI comparison widget; it keeps the Langfuse traces (`mode=live`) so every live run remains auditable.

## Comparison / Decision
Decide whether to fine-tune using `compare.py` with a positive delta threshold per track:
```bash
python compare.py --track structured --azure 05_eval/structured_azure_test.json --slm 05_eval/structured_slm_test.json --delta 0.01
python compare.py --track toolcall --azure 05_eval/toolcall_azure_test.json --slm 05_eval/toolcall_slm_test.json --delta 0.01
```
The script prints `FINE_TUNE=1` if the SLM trails Azure on any core metric by more than `delta`, logs a Langfuse event, and exits with status 10 to signal downstream automation.

## Conditional Fine-Tune (Unsloth QLoRA on 4×3090)
Run *only* when the compare step requests it.

### Strategy 1: Parallel Training (Recommended for small datasets)
Train both models simultaneously using 2 GPUs each. **Halves wallclock time** compared to sequential:
```bash
./train_parallel.sh
```
This launches:
- Structured model on GPUs 0,1
- Toolcall model on GPUs 2,3

Monitor with: `tail -f logs/train_structured_2gpu.log logs/train_toolcall_2gpu.log`

### Strategy 2: Sequential Training (Maximum per-model speed)
Train one model at a time using all 4 GPUs:
```bash
./train_sequential.sh
```

### Manual Training
Launch with `accelerate` (preferred) or `torchrun`:
```bash
# Single track with 4 GPUs
accelerate launch --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured

# Or with torchrun
torchrun --nproc_per_node=4 train_unsloth.py --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall
```

**Which strategy?** For datasets <10k examples, parallel training (2+2 GPUs) typically completes faster overall despite lower per-GPU efficiency. For larger datasets or if one track needs more attention, use sequential (4 GPUs per model).

Fixed hyperparameters: 8-bit load, LoRA `r=64`, `alpha=16`, `dropout=0.1`, `lr=2e-4`, `epochs=1`, `seq_len≈2048`. Training traces capture dataset paths, hyperparameters, adapter path, and multi-GPU strategy.

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
