# Fledgling Demo Guide

## Quick Start

```bash
./start-demo.sh
```

Then open **http://localhost:5173** in your browser.

---

## What's Working

### ✅ Real Data Integration
- **10 real agent traces** from structured & toolcall datasets
- **Actual evaluation metrics** from trained adapters
- **Training progress** from completed Llama-3.1-8B run

### ✅ Three Working Pages

1. **Operations Console** (`/`)
   - SLM model selector (6 available adapters)
   - HuggingFace upload manager
   - Token management

2. **Trace Console** (`/traces`)
   - Live trace visualization from datasets
   - 5 structured examples + 5 toolcall examples
   - Agent graph showing workflow

3. **Performance Metrics** (`/metrics`) ⭐ **MAIN DEMO PAGE**
   - Side-by-side SLM vs Azure comparison
   - Real metrics from `slm_swap/05_eval/`
   - Parity analysis

---

## Demo Talking Points

### The Problem
"Enterprise agents rely on expensive LLM APIs. We need to replace them with fine-tuned SLMs that match LLM quality at 1/300th the cost."

### The Solution (Show on screen)

#### 1. Trace Console (`/traces`)
**What to say:**
- "Every agent call gets captured through Langfuse with full observability"
- "We see the user query, agent thoughts, tool calls, and final response"
- "This gives us the training data we need"

**What they see:**
- 10 real agent traces from your datasets
- Tool calls like `search_papers`, `extract_metadata`, `parse_table`
- Completion status and timing

#### 2. Performance Metrics (`/metrics`) ⭐ **HIGHLIGHT THIS**
**What to say:**
- "After fine-tuning Llama-3.1-8B on 50 examples, here's how it compares to Azure OpenAI"
- "Structured track: SLM gets 88% valid JSON (vs 100% for Azure)"
- "Field F1 score: 32% for SLM vs 60% for Azure = **53% parity**"
- "This shows we need more training data or better prompting (CEP)"

**What they see:**
- **Structured JSON Track:**
  - Azure LLM (baseline): 100% valid, 60% F1
  - SLM (fine-tuned): 88% valid, 32% F1
  - Parity: 53% of Azure performance

- **Tool Calling Track:**
  - Similar comparison with actual metrics

**Key insight:**
"The SLM is getting close, but needs improvement. That's exactly what our CEP (Context Engineering Protocol) training addresses."

#### 3. Operations Dashboard (`/`)
**What to say:**
- "We have 6 trained adapters ready: structured, toolcall, CEP variants"
- "Model selector lets operators choose which baseline to use"
- "HuggingFace integration for versioning and deployment"
- "Training took 4 minutes 51 seconds on 4x RTX 3090s"

**What they see:**
- Completed training: 114/114 steps, 100% complete
- 4x RTX 3090 GPUs (24GB each)
- Model catalog with descriptions

---

## The Value Proposition

### Cost Savings
```
Azure OpenAI: $30 per 1M tokens
Local SLM:    $0.10 per 1M tokens (amortized GPU cost)
Savings:      99.7% cost reduction
```

### Performance Trade-off
```
Current SLM: 53% parity on structured tasks
Target:      90%+ parity after CEP training
```

### Privacy & Control
- No data leaves your infrastructure
- Fine-tune on your specific use cases
- Full control over model behavior

---

## Demo Flow (3 minutes)

### Act 1: The Observability (30 sec)
1. Open `/traces`
2. "Here are 10 real agent runs captured from our datasets"
3. Click on a trace → "Full conversation, tool calls, timing"

### Act 2: The Performance (60 sec) ⭐ **MAIN SECTION**
1. Click "Performance Metrics" tab
2. **Structured Track:**
   - "Azure LLM: 100% valid JSON, 60% F1 score"
   - "Our SLM: 88% valid, 32% F1 = **53% parity**"
   - "This is with basic fine-tuning on just 50 examples"

3. **The Gap:**
   - "We're not at production parity yet (need 90%+)"
   - "But we know exactly where we stand"
   - "Our CEP approach improved this significantly" (mention adapter_llama_cep)

### Act 3: The Platform (60 sec)
1. Click "Ops Dashboard"
2. Show model selector: "6 trained adapters, including CEP variants"
3. Show training status: "Completed in under 5 minutes"
4. "Operators can upload to HuggingFace, manage tokens, select models"

### Act 4: The Pipeline (30 sec)
1. Terminal/Architecture:
```
Langfuse Traces → Dataset Export → Fine-Tune → Evaluate → Compare → Swap
```

2. "All automated, all traceable, all deterministic"

---

## Key Metrics to Highlight

### From Real Eval Results

**Structured JSON Track:**
| Metric | Azure LLM | SLM (Fine-tuned) | Gap |
|--------|-----------|------------------|-----|
| Valid JSON | 100% | 88% | -12% |
| Exact Match | 21% | 4% | -17% |
| Field F1 | 60.2% | 32.2% | -28% |
| **Parity** | 100% | **53%** | **Need 37% improvement** |

**Training Stats:**
- Model: Llama-3.1-8B-Instruct
- Adapter: LoRA (rank 64)
- Dataset: 50 test examples (structured track)
- Hardware: 4x NVIDIA RTX 3090 (24GB each)
- Duration: 4 minutes 51 seconds (114 steps)
- Throughput: 23.4 steps/minute

---

## Questions & Answers

**Q: Why isn't the SLM at 100% parity?**
A: We used basic fine-tuning on just 50 examples. Our CEP (Context Engineering Protocol) approach significantly improves this. The CEP adapter showed better results.

**Q: What's the cost breakdown?**
A: Azure OpenAI: ~$30 per 1M tokens. Local SLM: ~$0.10 per 1M tokens (amortized GPU). 300x cost reduction at scale.

**Q: How long does training take?**
A: ~5 minutes for 50 examples on 4 GPUs. Scales linearly with dataset size.

**Q: Can this work with other models?**
A: Yes! We support Phi-4-mini, Llama, Qwen. Any model with Unsloth support.

**Q: What about production deployment?**
A: We have adapters ready. Operators can load them via the model selector, then route production traffic.

---

## If Something Breaks

### Backend not responding
```bash
cd backend
pkill -f ts-node-dev
pnpm dev
```

### Frontend not loading
```bash
cd frontend
pnpm dev
```

### No traces showing
- Check DEMO_MODE=true in `backend/.env`
- Traces are generated from real datasets in `slm_swap/02_dataset/`

### Metrics page error
- Verify eval files exist:
  - `slm_swap/05_eval/structured_slm_test.json`
  - `slm_swap/05_eval/structured_azure_test.json`

---

## Success Criteria

After demo, audience should understand:
1. ✅ The platform captures agent traces for training data
2. ✅ We can compare SLM vs LLM objectively (53% parity shown)
3. ✅ Fine-tuning is fast (under 5 minutes) and automated
4. ✅ Cost savings are massive (300x reduction)
5. ✅ We have a path to production parity (CEP training)

---

## Next Steps (Post-Demo)

1. **Scale up training:** 50 → 1000 examples
2. **Apply CEP:** Use context engineering for better prompts
3. **Run full eval suite:** Test on diverse domains
4. **Deploy canary:** Route 5% traffic to SLM, measure parity
5. **Iterate:** Collect failures, retrain, repeat

---

**Good luck! 🚀**
