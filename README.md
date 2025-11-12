# Fledgling

**Making task inference cheaper and more reliable**

Drop-in observability and fine-tuning tools for AI agents. Instrument your agents with 10 lines of code (soon a click of a button), get full trace visibility, and fine-tune local SLMs (like Phi-4-mini) to replace expensive LLMs.

> **⚠️ Note:** The TypeScript frontend/backend and Python fine-tuning pipeline are **not yet fully integrated**. They can be run independently but don't communicate automatically. Integration is in progress.

## What's Here

### TypeScript Application (Main Platform)
- **`backend/`** - Express.js API for agent management, trace fetching, model comparison, and SLM selection
- **`frontend/`** - React UI for viewing agents, traces, side-by-side comparisons, and SLM selector dashboard
- **`demo-user/`** - Example Mastra agent with Fledgling instrumentation

### Python Pipeline (Fine-Tuning - Standalone)
- **`python-pipeline/slm_swap/`** - Evaluation and training scripts
  - `prepare_data.py` - Dataset preparation
  - `eval.py` - Model evaluation harness
  - `compare.py` - Decision logic for fine-tuning
  - `train_unsloth.py` - QLoRA fine-tuning with Unsloth (multi-GPU support)

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
         ⚠️ NOT YET CONNECTED ⚠️
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
| Model evaluation | ✅ Working | Azure and SLM eval tested |
| Fine-tuning pipeline | ✅ Working | Unsloth QLoRA multi-GPU ready |
| **UI → Python integration** | ❌ Not integrated | Manual export/import required |
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
- Evaluate Azure LLM vs local SLM
- Decide if fine-tuning is needed
- Train QLoRA adapters (multi-GPU)
- Re-evaluate fine-tuned models

**To connect them yourself:**
1. Export traces from Langfuse as JSONL
2. Transform to training format
3. Run Python pipeline
4. Load adapters in your agent code manually

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
