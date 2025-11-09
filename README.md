# Fledgling

**Making task inference cheaper and more reliable**

Drop-in observability and fine-tuning tools for AI agents. Instrument your Mastra agents with 10 lines of code, get full trace visibility, and optionally fine-tune local SLMs to replace expensive LLMs.

> **⚠️ Note:** The TypeScript frontend/backend and Python fine-tuning pipeline are **not yet fully integrated**. They can be run independently but don't communicate automatically. Integration is in progress.

## What's Here

### TypeScript Application (Main Platform)
- **`backend/`** - Express.js API for agent management, trace fetching, and model comparison
- **`frontend/`** - React UI for viewing agents, traces, and side-by-side comparisons
- **`demo-user/`** - Example Mastra agent with Fledgling instrumentation

### Python Pipeline (Fine-Tuning - Standalone)
- **`python-pipeline/slm_swap/`** - Evaluation and training scripts
  - `prepare_data.py` - Dataset preparation
  - `eval.py` - Model evaluation harness
  - `compare.py` - Decision logic for fine-tuning
  - `train_unsloth.py` - QLoRA fine-tuning with Unsloth

## Quick Start

### Prerequisites
- **For TypeScript app:** Node.js 18+, pnpm 8+
- **For Python pipeline:** Python 3.11+, CUDA GPU (24GB VRAM recommended)
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

### Setup Python Pipeline (Optional)

1. **Install Python dependencies:**
   ```bash
   cd python-pipeline
   pip install -r requirements.txt
   ```

2. **Download a base SLM:**
   ```bash
   huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
     --local-dir slm_swap/models/qwen2.5-7b-instruct
   ```

3. **Configure environment variables:**
   ```bash
   export AZURE_ENDPOINT="https://<endpoint>/openai/deployments/<deployment>"
   export AZURE_API_KEY="..."
   export AZURE_API_VERSION="2024-02-15-preview"
   export SLM_MODEL_PATH="slm_swap/models/qwen2.5-7b-instruct"
   # Reuse Langfuse credentials from above
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

   # Fine-tune if needed (exits with code 10)
   python train_unsloth.py --track structured \
     --train 02_dataset/structured/train.jsonl \
     --val 02_dataset/structured/val.jsonl \
     --out 04_ft/adapter_structured
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

### Fine-Tune Models (Manual Process)

Currently, you need to:
1. Export traces from Langfuse manually
2. Transform into training format (or use provided datasets)
3. Run Python pipeline scripts
4. Load trained adapters manually in your agents

**Future:** This will be automated end-to-end.

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
└──────────────────────┘        └──────────────────────────┘
         ⚠️ NOT YET CONNECTED ⚠️
```

## Current Integration Status

| Feature | Status | Notes |
|---------|--------|-------|
| Agent instrumentation | ✅ Working | Mastra agents auto-trace to Langfuse |
| Trace viewing in UI | ✅ Working | Full trace detail in React app |
| Agent registration | ✅ Working | Demo-user auto-registers |
| Dataset preparation | ✅ Working | Python script functional |
| Model evaluation | ✅ Working | Both Azure and SLM eval tested |
| Fine-tuning pipeline | ✅ Working | Unsloth QLoRA implementation ready |
| **UI → Python integration** | ❌ Not integrated | Manual export/import required |
| **Auto-trigger training** | ❌ Not integrated | Must run Python scripts manually |
| **Load trained models in UI** | ❌ Not integrated | No adapter loading in TypeScript |
| **Cost comparison** | ⚠️ Partial | Calculator exists, needs trained models |

## What Works Independently

**TypeScript App:**
- Run agents with full observability
- View traces in UI
- Compare agent configurations
- See token usage and costs

**Python Pipeline:**
- Prepare training datasets
- Evaluate Azure LLM vs local SLM
- Decide if fine-tuning is needed
- Train QLoRA adapters
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
- Langfuse SDK, OpenAI client
- bitsandbytes, datasets, TRL

## Roadmap

- [ ] Automated trace export from TypeScript backend
- [ ] Trigger Python pipeline via API/webhooks
- [ ] Load trained adapters in Mastra agents
- [ ] Cost comparison visualization with real models
- [ ] One-click deployment to Azure ML/HuggingFace
- [ ] Framework adapters (LangChain, CrewAI, etc.)

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
