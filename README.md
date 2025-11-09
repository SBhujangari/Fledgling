# Fledgling
Making task inference cheaper and more reliable with a local Phi-4-mini SLM + Azure LLM baseline.

## Local SLM Download (Phi-4-mini)
```bash
huggingface-cli download microsoft/phi-4-mini \
  --local-dir slm_swap/models/phi-4-mini
```
The `slm_client` loads this checkpoint with 8-bit quantization, `device_map="auto"`, and `torch_dtype=torch.bfloat16`, so it automatically shards across four RTX 3090s when available. Evaluation uses batched inference (default batch_size=8) to process multiple examples in parallel across all 4 GPUs for maximum throughput.

## Environment Variables
```bash
export SLM_MODEL_PATH="slm_swap/models/phi-4-mini"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
# optional local smoke tests
# export LANGFUSE_DISABLED=1

export AZURE_ENDPOINT="https://<resource>.openai.azure.com/openai/deployments/<deployment>"
export AZURE_DEPLOYMENT="<deployment>"
export AZURE_API_KEY="..."
export AZURE_API_VERSION="2024-05-01-preview"
```
The Azure endpoint must follow the conventional `/openai/deployments/<deployment>` form or an equivalent `models/chat/completions` endpoint that your resource exposes.

## Multi-GPU Fine-Tuning (Unsloth QLoRA)

### Quick Start
```bash
cd slm_swap

# Parallel training (2+2 GPUs) - recommended for small datasets, halves wallclock time
./train_parallel.sh

# Sequential training (4 GPUs per model) - maximum per-model speed
./train_sequential.sh
```

### Manual Launch
Launch via Accelerate (preferred) or Torchrun:
```bash
# Accelerate (4 processes, ZeRO-2 via accelerate_config/deepspeed_zero2.json)
accelerate launch --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured

# Torchrun equivalent
torchrun --nproc_per_node=4 train_unsloth.py --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall
```
`train_unsloth.py` wires gradient checkpointing, `ddp_find_unused_parameters=False`, and default DeepSpeed ZeRO-2 config (`slm_swap/accelerate_config/deepspeed_zero2.json`) so multi-GPU utilization is automatic.

## SLM Selector (Dashboard)
Operators can now choose which SLM checkpoint becomes the default baseline for the next fine-tuning pass—either a Hugging Face base model or one of our local adapters.

1. Edit `slm_swap/model_catalog.json` to add/remove entries (each entry can point at an HF repo id or local folder such as `slm_swap/04_ft/adapter_structured`).
2. Start `backend` + `frontend` (both `npm run dev`). The Operations Console shows a **SLM Fine-Tune Selector** panel where you can pick any available model (unavailable entries are disabled).
3. Selection is persisted to `slm_swap/model_selection.json`; backend consumers can read it directly or via HTTP.

API surface:

```bash
# List catalog + current selection
curl http://localhost:4000/api/slm/models

# Update selection
curl -X POST http://localhost:4000/api/slm/select \
  -H "Content-Type: application/json" \
  -d '{"modelId":"hf-phi-4-mini"}'
```

Training scripts can read the JSON to decide which checkpoint to fine-tune or evaluate next.

## Hugging Face Upload Helper
GitHub blocks binary blobs larger than 100 MB, so push adapters, eval logs, or datasets to a private Hugging Face repo until you are ready to make them public.

1. Store a **write-scoped** Hugging Face token once (either via dashboard or API). Tokens are saved in `backend/.hf_token`:

   ```bash
   curl -X POST http://localhost:4000/api/hf/token \
     -H "Content-Type: application/json" \
     -d '{"token":"hf_xxx"}'
   ```

2. Start the backend + frontend (`npm run dev` in each folder). The Operations Console exposes a token manager and an **Hugging Face Upload** panel where operators can pick target folders, tweak the commit message, and ship artifacts with one click—no extra CLI login required.

The backend exposes the same upload workflow over HTTP so you can call it from other systems:

```bash
curl -X POST http://localhost:4000/api/hf/upload \
  -H "Content-Type: application/json" \
  -d '{
    "repoId": "your-username/slm-adapters",
    "paths": [
      "slm_swap/04_ft/adapter_structured",
      "slm_swap/04_ft/adapter_toolcall"
    ],
    "private": true,
    "autoSubdir": true,
    "commitMessage": "sync adapters via API"
  }'
```

Under the hood both the CLI and HTTP endpoints call `slm_swap/hf_upload.py`, so you can still run it manually when scripting larger workflows:

```bash
huggingface-cli login  # or export HUGGING_FACE_HUB_TOKEN
python slm_swap/hf_upload.py \
  slm_swap/04_ft/adapter_structured \
  slm_swap/04_ft/adapter_toolcall \
  --repo-id your-username/slm-adapters \
  --private \
  --auto-subdir
```

Use `--repo-type dataset` to upload JSONL corpora instead of model weights, `--path-in-repo <subdir>` to pin the destination folder, and rerun the script any time you need to sync large assets with the Hub.

## TODO / Upcoming
- [ ] Auto-trigger training scripts based on the selected SLM (dashboard now captures the choice between Hugging Face bases and local fine-tuned adapters).

## Repo Map
- `slm_swap/README.md` — full workflow (datasets, baseline evals, comparisons, fine-tune loop).
- `slm_swap/agent.md` — operator playbook detailing the evaluation-first plan.
- `slm_swap/accelerate_config/*.yaml|json` — launcher templates for 4×3090 training.
- `requirements.txt` — install once from repo root (`deepspeed`, `unsloth`, `trl`, etc.).

See `slm_swap/README.md` for end-to-end instructions covering dataset prep, Langfuse expectations, evaluations, and decision rules.

## Future Roadmap — Model Graders vs Teacher–Student
We may later layer in OpenAI’s model-grader-driven reinforcement fine-tuning (RFT) workflow that relies on explicit reward functions to shape reasoning behavior (inspiration: [OpenAI Cookbook example](https://cookbook.openai.com/examples/reinforcement_fine_tuning)). That approach differs from today’s teacher/student plan (`AGENT.md`) which prioritizes swapping in a local SLM, matching teacher metrics track-by-track, and only running Unsloth QLoRA when the SLM underperforms. For now this section is informational only—no grader-based automation is scheduled for implementation yet, but keeping the contrast documented helps if we decide to expand the roadmap.

## Roadmap — Reinforcement Fine‑Tuning (RFT)
- Default posture: keep SFT-only for current structured JSON and tool‑calling tracks; RFT is not required to achieve a fast, automated LLM→SLM swap.
- When to consider RFT: SFT saturates below target metrics, or new tasks demand complex reasoning/subjective behavior with stable, programmatic reward signals tied to business outcomes.
- Prerequisites: SFT warm‑start; trajectory capture; observation masking; unified reward interface from existing objective metrics; KL‑regularized updates to a reference; strict, deterministic acceptance gates; full Langfuse traceability.
- Migration path: warm‑start from SFT adapters; define rewards from existing metrics (e.g., json_valid_rate, exact_match_rate, name/args correctness); run short, tightly KL‑regularized RL phases with frequent holdout evaluation; block deployment unless objective thresholds are met or exceeded.
- Risks: reward hacking, higher compute/ops cost, regressions on objective metrics; mitigate with KL control, no‑regression gates, and frequent holdout checks.
- Status: not in v0/v1; staged for later if SFT caps out or new reasoning‑heavy tasks are introduced.
