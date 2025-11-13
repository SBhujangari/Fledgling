# Fledgling

**Agent-parity + swap kit: prove your specialist SLM beats the hosted LLM, then flip traffic**

Fledgling isn't a generic "fine-tuning platform." We capture live agent traces, build eval harnesses around your exact tools and JSON schemas, auto-curate the right datasets, train a specialist SLM (on whatever trainer you prefer), and ship a swap-readiness report plus a deterministic drop-in runner. Think: **measure → prove parity → ship.**

### How we're different
- **Unit of value:** Everyone else is model-centric ("let us fine-tune/RL your model"). Fledgling is agent-centric—parity on your real workflow (tool calls, JSON I/O, routers) with acceptance tests and canary rollout built in.
- **What we deliver:** Others hand you a fine-tuned model + generic evals. Fledgling hands you a ParityBench report (tool-use accuracy, schema conformance, latency/cost deltas), auto-curated datasets from your traces, and a swap kit (schema enforcer, fallback router, deterministic seeds/logging).
- **Training backend stance:** Others pull you into their trainer/cloud. Fledgling is vendor-agnostic—we can call OpenPipe/ART, Predibase RFT, H2O Studio, Databricks/MosaicML, or your Unsloth/QLoRA rig. We own the measurement, dataset ops, and swap logic.
- **Determinism & governance:** MLOps stacks exist, but none are purpose-built for agent-parity gating. We enforce deterministic guardrails (schema + tool contracts + seeds), traceable provenance, and pass/fail gates tied to your acceptance criteria before we reroute live traffic.

> **⚠️ Note:** The TypeScript frontend/backend and Python fine-tuning pipeline are **not yet fully integrated**. They can be run independently but don't communicate automatically. Integration is in progress.

## What's Here

### TypeScript Application (Main Platform)
- **`backend/`** - Express.js API for agent management, trace fetching, model comparison, and SLM selection
- **`frontend/`** - React UI for viewing agents, traces, side-by-side comparisons, and SLM selector dashboard
- **`demo-user/`** - Example Mastra agent with Fledgling instrumentation

### Python Pipeline (Fine-Tuning - Standalone)
- **`python-pipeline/slm_swap/`** - Evaluation and training scripts
  - `dummy_agent_workflow.py` - Generate Langfuse-style traces without a live agent
  - `langfuse_dataset.py` - Harvest Langfuse traces into structured/tool-call splits
  - `prepare_data.py` - Dataset preparation
  - `eval.py` - Model evaluation harness
  - `compare.py` - Decision logic for fine-tuning
  - `train_unsloth.py` - QLoRA fine-tuning with Unsloth (multi-GPU support)
  - `run_dummy_pipeline.py` - One-click dummy workflow → dataset → dry-run FT tester

## Quick Start

### Prerequisites
- **For TypeScript app:** Node.js 18+, pnpm 8+
- **For Python pipeline:** Python 3.11+, CUDA GPU (24GB VRAM recommended, optimized for 4×RTX 3090)
- **For both:** Langfuse account (cloud or self-hosted)

### Setup TypeScript Application

1. **Install dependencies:**
   ```bash
   pnpm install
   ```

2. **Configure environment variables:**
   ```bash
   # Backend (.env in backend/)
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   PORT=3000

   # Demo user (.env in demo-user/)
   OPENAI_API_KEY=sk-...
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

3. **Run the app:**
   ```bash
   pnpm dev  # Runs backend (3000), frontend (5173), demo-user (3001)
   ```

### Setup Python Pipeline

1. **Install Python dependencies:**
   ```bash
   cd python-pipeline
   pip install -r requirements.txt
   ```

2. **Download a base SLM (Phi-4-mini recommended):**
   ```bash
   huggingface-cli download microsoft/phi-4-mini \
     --local-dir slm_swap/models/phi-4-mini
   ```
   The `slm_client` loads this checkpoint with 8-bit quantization, `device_map="auto"`, and `torch_dtype=torch.bfloat16`, automatically sharding across available GPUs. Evaluation uses batched inference (default batch_size=8) for maximum throughput.

3. **Configure environment variables:**
   ```bash
   export SLM_MODEL_PATH="slm_swap/models/phi-4-mini"
   export LANGFUSE_PUBLIC_KEY="..."
   export LANGFUSE_SECRET_KEY="..."
   export LANGFUSE_HOST="https://cloud.langfuse.com"

   export AZURE_ENDPOINT="https://<resource>.openai.azure.com/openai/deployments/<deployment>"
   export AZURE_DEPLOYMENT="<deployment>"
   export AZURE_API_KEY="..."
   export AZURE_API_VERSION="2024-05-01-preview"
   ```

4. **Run the pipeline:**
   ```bash
   cd slm_swap

   # Prepare dataset
   python prepare_data.py --dataset-path <path-to-dataset>

   # Run evaluations
   python eval.py --track structured --model-kind azure --split test
   python eval.py --track structured --model-kind slm --split test

   # Compare and decide
   python compare.py --track structured \
     --azure 05_eval/structured_azure_test.json \
     --slm 05_eval/structured_slm_test.json \
     --delta 0.01

   # Fine-tune if needed
   python train_unsloth.py --track structured \
     --train 02_dataset/structured/train.jsonl \
     --val 02_dataset/structured/val.jsonl \
     --out 04_ft/adapter_structured
   ```

## Dummy Workflow & Pipeline Smoke Test

Need to prove the Langfuse → dataset → fine-tune loop works without touching Azure or a GPU? Run the fully offline harness:

```bash
python python-pipeline/slm_swap/run_dummy_pipeline.py --clean --count 12
```

This command:
1. Synthesizes dummy agent traces whose prompts/completions match our eval datasets (`dummy_agent_workflow.py`).
2. Converts those traces into structured/tool-call JSONLs via `langfuse_dataset.py --trace-json ...`.
3. Executes `train_unsloth.py --dry-run` for both tracks so the whole loop is validated without loading a model.
4. Writes `slm_swap/logs/finetune_progress.json`, which powers the frontend tuning dashboard via `/api/training/status`.

To push the dummy traces into a real Langfuse workspace, rerun the generator with `--emit-to-langfuse` and real `LANGFUSE_*` vars, then re-run `langfuse_dataset.py` without `--trace-json`.

## Langfuse Trace Console (Original Fledgling UI)

Need a quick view into raw Langfuse spans before kicking off the pipeline?

```bash
cd Fledgling
npm install          # once
npm run dev          # installs backend/frontend (via postinstall) then starts http://localhost:5173
```

The console automatically loads `/api/traces`, summarizes runs/observations, and links directly to each Langfuse trace so you can audit data before exporting it into the fine-tuning pipeline.

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
# Accelerate (4 processes, ZeRO-2)
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
`train_unsloth.py` configures gradient checkpointing, `ddp_find_unused_parameters=False`, and DeepSpeed ZeRO-2 for automatic multi-GPU utilization.

## SLM Selector (Dashboard)
Operators can choose which SLM checkpoint becomes the default baseline for the next fine-tuning pass—either a Hugging Face base model or a local adapter.

1. Edit `slm_swap/model_catalog.json` to add/remove entries (each entry can point at an HF repo id or local folder like `slm_swap/04_ft/adapter_structured`).
2. Start `backend` + `frontend` (`npm run dev`). The Operations Console shows a **SLM Fine-Tune Selector** panel.
3. Selection persists to `slm_swap/model_selection.json`; backend consumers can read it via HTTP.

**API:**
```bash
# List catalog + current selection
curl http://localhost:4000/api/slm/models

# Update selection
curl -X POST http://localhost:4000/api/slm/select \
  -H "Content-Type: application/json" \
  -d '{"modelId":"hf-phi-4-mini"}'
```

## Hugging Face Upload Helper
GitHub blocks binaries >100 MB, so push adapters, eval logs, or datasets to Hugging Face.

1. **Store a write-scoped token** (via dashboard or API):
   ```bash
   curl -X POST http://localhost:4000/api/hf/token \
     -H "Content-Type: application/json" \
     -d '{"token":"hf_xxx"}'
   ```

2. **Upload via UI or API:**
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
       "commitMessage": "sync adapters"
     }'
   ```

3. **Manual CLI:**
   ```bash
   huggingface-cli login
   python slm_swap/hf_upload.py \
     slm_swap/04_ft/adapter_structured \
     slm_swap/04_ft/adapter_toolcall \
     --repo-id your-username/slm-adapters \
     --private \
     --auto-subdir
   ```

4. **Automatic promotion after eval** – `slm_swap/compare_llm_slm.py` can now publish an adapter as soon as it meets your acceptance gap. Example (uploads only if the SLM matches or beats the LLM on the selected split):

   ```bash
   python slm_swap/compare_llm_slm.py \
     --track structured \
     --split eval100 \
     --adapter slm_swap/04_ft/adapter_structured \
     --auto-upload-repo kineticdrive/test \
     --auto-upload-repo-type space \
     --auto-upload-path-in-repo adapter_structured \
     --auto-upload-auto-subdir \
     --auto-upload-gap-threshold 0.0
   ```

   Pass `--auto-upload-token hf_xxx` (or rely on `HUGGING_FACE_HUB_TOKEN`), tweak `--auto-upload-commit-message`, or raise the `--auto-upload-gap-threshold` if you only want to ship once the local model leads by a margin.

## Usage

### Instrument Your Agent

```typescript
import { withMastraTracing } from '@fledgling/tracer';
import { Agent } from '@mastra/core';

const agent = new Agent({
  name: 'my-agent',
  model: openai('gpt-4'),
  tools: [myTool]
});

// Wrap with Fledgling observability
const tracedAgent = withMastraTracing(agent);

// Use normally - traces automatically sent to Langfuse
const result = await tracedAgent.generate('Hello world');
```

### View Traces in UI

1. Open http://localhost:5173
2. See registered agents
3. View execution traces with full detail
4. (Comparison features ready, awaiting trained models)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR AGENT APP                        │
│                 + withMastraTracing()                    │
└────────────────────────┬─────────────────────────────────┘
                         │ OpenTelemetry → Langfuse
                         ▼
┌──────────────────────────────────────────────────────────┐
│                      LANGFUSE                            │
│              (Shared observability layer)                │
└──────────────┬────────────────────────────┬──────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────┐        ┌──────────────────────────┐
│ TYPESCRIPT APP       │        │ PYTHON PIPELINE          │
│ • Frontend UI        │        │ • prepare_data.py        │
│ • Backend API        │        │ • eval.py                │
│ • Trace viewing      │        │ • compare.py             │
│ • Agent management   │        │ • train_unsloth.py       │
│ • SLM selector       │        │ • HF upload              │
└──────────────────────┘        └──────────────────────────┘
         Langfuse dataset exporter pipes traces → Python splits
```

## Current Integration Status

| Feature | Status | Notes |
|---------|--------|-------|
| Agent instrumentation | ✅ Working | Mastra agents auto-trace to Langfuse |
| Trace viewing in UI | ✅ Working | Full trace detail in React app |
| Agent registration | ✅ Working | Demo-user auto-registers |
| SLM selector dashboard | ✅ Working | Choose base model or adapter |
| HF upload helper | ✅ Working | Token management + upload API |
| Dataset preparation | ✅ Working | Python script functional |
| Langfuse → dataset export | ✅ Working | `langfuse_dataset.py` builds structured/tool-call splits from traces |
| Dummy pipeline smoke test | ✅ Working | `run_dummy_pipeline.py` exercises traces → dataset → dry-run FT |
| Model evaluation | ✅ Working | Azure and SLM eval tested |
| Fine-tuning pipeline | ✅ Working | Unsloth QLoRA multi-GPU ready |
| **UI → Python integration** | ⚠️ Partial | Dataset export is automated; orchestration + adapter loading still manual |
| **Auto-trigger training** | ❌ Not integrated | Must run Python scripts manually |
| **Load trained models in UI** | ❌ Not integrated | No adapter loading in TypeScript |
| **Cost comparison** | ⚠️ Partial | Calculator exists, needs trained models |

## What Works Independently

**TypeScript App:**
- Run agents with full observability
- View traces in UI
- Compare agent configurations
- Select SLM checkpoint for fine-tuning
- Upload artifacts to Hugging Face
- See token usage and costs

**Python Pipeline:**
- Prepare training datasets
- Harvest Langfuse traces into structured/tool-call JSONLs
- Generate offline dummy traces (`dummy_agent_workflow.py`) and convert them via `langfuse_dataset.py --trace-json`
- Evaluate Azure LLM vs local SLM
- Decide if fine-tuning is needed
- Train QLoRA adapters (multi-GPU)
- Re-evaluate fine-tuned models
- Dry-run the training loop (`train_unsloth.py --dry-run`) for fast wiring tests

**To connect them yourself (until we wire the UI triggers):**
1. Run `python python-pipeline/slm_swap/langfuse_dataset.py --agent-id <id> --output-root python-pipeline/slm_swap/02_dataset`
2. Run the Python pipeline (eval → compare → optional train)
3. Load adapters in your agent code manually

## Tech Stack

**TypeScript:**
- React 19, Vite, Tailwind, Radix UI
- Express, TypeScript, LowDB
- OpenTelemetry, Langfuse
- Mastra agent framework

**Python:**
- PyTorch, Transformers, Unsloth
- Langfuse SDK, OpenAI/Azure clients
- bitsandbytes, datasets, TRL, DeepSpeed

## Repo Map
- `slm_swap/README.md` — full workflow (datasets, baseline evals, comparisons, fine-tune loop)
- `slm_swap/agent.md` — operator playbook detailing the evaluation-first plan
- `slm_swap/accelerate_config/*.yaml|json` — launcher templates for multi-GPU training
- `requirements.txt` — install from repo root (`deepspeed`, `unsloth`, `trl`, etc.)

## Roadmap

### Near Term
- [ ] Auto-trigger training scripts based on selected SLM
- [ ] Automated trace export from TypeScript backend
- [ ] Trigger Python pipeline via API/webhooks
- [ ] Load trained adapters in Mastra agents
- [ ] Cost comparison visualization with real models

### Future
- [ ] One-click deployment to Azure ML/HuggingFace
- [ ] Framework adapters (LangChain, CrewAI, etc.)
- [ ] Reinforcement Fine-Tuning (RFT) integration

## Reinforcement Fine-Tuning (RFT) - Future Consideration

**Default posture:** SFT-only for structured JSON and tool-calling tracks; RFT not required for fast LLM→SLM swap.

**When to consider:** SFT saturates below target metrics, or tasks demand complex reasoning with stable reward signals.

**Prerequisites:** SFT warm-start; trajectory capture; unified reward interface; KL-regularized updates; strict acceptance gates; Langfuse traceability.

**Status:** Not in v0/v1; staged for later if SFT caps out or reasoning-heavy tasks emerge.

## Known Issues

- Python pipeline requires manual execution (no auto-trigger)
- Trained models not automatically available to TypeScript app
- No adapter loading in demo-user example
- Cost calculator needs actual fine-tuned model metrics

## Contributing

This is a hackathon project under active development. The architecture is designed for full integration—we just haven't wired everything together yet.

If you want to help connect the pieces, check out:
- `backend/src/routes/` - Where we'd add Python pipeline triggers
- `python-pipeline/slm_swap/` - Standalone scripts ready to be called
- `demo-user/src/agent.ts` - Example of loading adapters manually

## License

MIT
