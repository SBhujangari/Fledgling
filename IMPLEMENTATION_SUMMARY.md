# Implementation Summary: LLM → SLM Function Calling Comparison Framework

**Date:** November 8, 2025
**Status:** ✅ Complete - Ready for Execution

---

## Executive Summary

We have successfully implemented a comprehensive comparison framework to evaluate two approaches for fine-tuning small language models (SLMs) for function calling:

1. **Fledgling Approach** (Current): Phi-4-mini with Azure OpenAI comparison and decision gate
2. **Paper Approach** (FUNCTION_CALLING_FINETUNE.md): Unsloth-based training with Hermes dataset

The framework supports **iterative experimentation** across four key capabilities:
- ✅ Single-turn function calling
- ✅ Multi-turn conversations
- ✅ Structured JSON outputs
- ✅ Diverse domain coverage

**Goal:** Prove that SLM fine-tuning can match or surpass LLM (Azure GPT) performance with ≤10% delta.

---

## What Was Implemented

### 1. Docker Containerization (Both Approaches)

#### Fledgling Container
- **Location:** `docker/fledgling/`
- **Base:** NVIDIA CUDA 12.1.0 + cuDNN 8
- **Purpose:** Run existing Fledgling pipeline in isolated environment
- **GPU Allocation:** All 4 GPUs (dynamic)
- **Key Files:**
  - `Dockerfile` - Container definition
  - `docker-compose.yml` - Orchestration
  - `entrypoint.sh` - Environment setup
  - `README.md` - Usage guide

#### Paper Approach Containers (2x)
- **Location:** `docker/unsloth-paper/`
- **Base:** Official `unsloth/unsloth:latest`
- **Purpose:** Run paper's approach with both Phi-4-mini and Llama-3.1-8B in parallel
- **GPU Allocation:**
  - `paper-approach-phi`: GPUs 0,1 (Phi-4-mini)
  - `paper-approach-llama`: GPUs 2,3 (Llama-3.1-8B)
- **Key Files:**
  - `Dockerfile` - Unsloth-based container
  - `docker-compose.yml` - Dual-container setup
  - `scripts/train_function_call.py` - Training script (paper method)
  - `scripts/eval_function_call.py` - Evaluation script (paper method)
  - `README.md` - Usage guide

---

### 2. Dataset Preparation

#### Hermes Function Calling v1 Dataset
- **Script:** `paper_approach/prepare_hermes_dataset.py`
- **Features:**
  - Downloads from `NousResearch/hermes-function-calling-v1`
  - Stratified sampling by complexity
  - Configurable splits (train/val/test)
  - Fallback to synthetic examples if download fails
- **Phase 1 Configuration:**
  - Train: 100 examples
  - Val: 50 examples
  - Test: 50 examples
- **Phase 2 Configuration:**
  - Train: 1000 examples
  - Val: 200 examples
  - Test: 200 examples

---

### 3. Specialized Test Suites

#### Single-Turn Test Suite
- **Script:** `comparison_framework/test_single_turn.py`
- **Purpose:** Test direct query → function call mappings
- **Categorization:**
  - **Simple:** 1 tool, required params only (20 examples)
  - **Moderate:** 1 tool, optional params (20 examples)
  - **Complex:** Multiple tools or nested params (10 examples)
- **Total:** 50 examples

#### Multi-Turn Test Suite
- **Script:** `comparison_framework/test_multi_turn.py`
- **Purpose:** Test complex conversations with context
- **Categorization:**
  - **Clarification:** User corrects/refines request (10 examples)
  - **Sequential:** Step-by-step tool calls (10 examples)
  - **Context-dependent:** Later calls depend on earlier context (10 examples)
- **Total:** 30 examples (if available in dataset)

#### Domain Diversity Test Suite
- **Script:** `comparison_framework/test_domain_diversity.py`
- **Purpose:** Test performance across diverse domains
- **Domains Detected:**
  - Weather, Search, E-commerce, Calendar, Maps
  - Communication, Finance, Database, Files, Social
- **Configuration:** 5 examples per domain (balanced sampling)
- **Total:** 50 examples

---

### 4. Unified Evaluation Framework

#### Comparison Script
- **Script:** `comparison_framework/compare_approaches.py`
- **Features:**
  - Loads results from both Fledgling and Paper approaches
  - Compares across all 4 capabilities
  - Generates comprehensive report (JSON + CSV)
  - Calculates delta to Azure LLM baseline
- **Metrics Tracked:**
  - Valid JSON rate
  - Exact match rate
  - Field precision/recall/F1
  - Name match rate (tool calling)
  - Domain-level performance

#### Evaluation Outputs
- **JSON Report:** `comparison_framework/reports/phase1_comparison.json`
- **CSV Exports:** `comparison_framework/reports/csv/`
  - `single_turn.csv`
  - `json_quality.csv`
  - `llm_comparison.csv`

---

### 5. Training Scripts (Paper Approach)

#### Training Configuration
- **Script:** `docker/unsloth-paper/scripts/train_function_call.py`
- **Features:**
  - Supports both Phi-4-mini and Llama-3.1-8B-Instruct
  - Unsloth FastLanguageModel with 4-bit quantization
  - LoRA fine-tuning (configurable rank/alpha/dropout)
  - Automatic metadata generation with version ID
  - Langfuse integration (optional)

#### Hyperparameters (Paper Defaults)
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Epochs | 3 |
| Max seq length | 2048 |
| Precision | BF16 |

#### Outputs
- **Phi-4-mini:** `paper_approach/adapters/phi_hermes_lora/`
- **Llama-3.1-8B:** `paper_approach/adapters/llama_hermes_lora/`
- Each contains:
  - `adapter_model.safetensors` (~100-200 MB)
  - `adapter_config.json`
  - `metadata.json` (training context + metrics)
  - Tokenizer files

---

### 6. Evaluation Scripts (Paper Approach)

#### Evaluation Script
- **Script:** `docker/unsloth-paper/scripts/eval_function_call.py`
- **Features:**
  - Loads base model + adapter
  - Greedy decoding (deterministic)
  - JSON extraction and validation
  - Per-example metrics tracking
  - Detailed output logging

#### Metrics Computed
1. **valid_json_rate:** Percentage of parseable JSON outputs
2. **name_match_rate:** Tool name accuracy
3. **args_exact_match_rate:** Perfect argument match
4. **args_field_precision/recall/f1:** Per-field accuracy

---

### 7. Master Orchestration

#### Automated Experiment Script
- **Script:** `run_phase1_experiment.sh`
- **Purpose:** End-to-end Phase 1 execution
- **Steps:**
  1. Check prerequisites (Docker, GPU, .env)
  2. Prepare Hermes dataset (100/50/50 split)
  3. Create specialized test splits
  4. Build Docker images
  5. Train both models in parallel (Phi + Llama)
  6. Evaluate on all test suites
  7. Generate comparison report

#### Execution Time
- **Setup:** ~5 minutes
- **Training:** ~60-90 minutes (parallel)
- **Evaluation:** ~10-15 minutes
- **Total:** ~90 minutes

#### Usage
```bash
cd /home/gabriel/Desktop/AI_ATL25
./run_phase1_experiment.sh
```

---

### 8. Comprehensive Documentation

#### RUN_EXPERIMENTS.md
- **Purpose:** Step-by-step experimental guide
- **Contents:**
  - Phase 0: Environment setup
  - Phase 1: Small-scale iterative validation (100 examples)
  - Phase 2: Scale up (1000 examples, if Phase 1 succeeds)
  - Phase 3: Production deployment
  - Success criteria for each phase
  - Troubleshooting guide

#### README Files
- `docker/fledgling/README.md` - Fledgling Docker usage
- `docker/unsloth-paper/README.md` - Paper approach Docker usage
- `FUNCTION_CALLING_FINETUNE.md` - Original paper methodology (existing)

---

## File Structure (New Additions)

```
AI_ATL25/
├── docker/
│   ├── fledgling/                     # NEW: Fledgling containerization
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── entrypoint.sh
│   │   └── README.md
│   │
│   └── unsloth-paper/                 # NEW: Paper approach containerization
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── entrypoint.sh
│       ├── README.md
│       └── scripts/
│           ├── train_function_call.py
│           └── eval_function_call.py
│
├── paper_approach/                    # NEW: Paper approach implementation
│   ├── prepare_hermes_dataset.py      # Dataset preparation
│   ├── datasets/                      # Hermes dataset splits
│   │   ├── hermes_train.jsonl
│   │   ├── hermes_val.jsonl
│   │   ├── hermes_test.jsonl
│   │   └── dataset_stats.json
│   ├── adapters/                      # Fine-tuned adapters
│   │   ├── phi_hermes_lora/
│   │   └── llama_hermes_lora/
│   ├── eval_results/                  # Evaluation outputs
│   │   ├── phi_single_turn.json
│   │   ├── llama_single_turn.json
│   │   ├── phi_domain_diverse.json
│   │   └── llama_domain_diverse.json
│   └── logs/                          # Training logs
│
├── comparison_framework/              # NEW: Unified comparison framework
│   ├── compare_approaches.py          # Main comparison script
│   ├── test_single_turn.py            # Single-turn test suite
│   ├── test_multi_turn.py             # Multi-turn test suite
│   ├── test_domain_diversity.py       # Domain diversity test suite
│   ├── test_splits/                   # Test split outputs
│   │   ├── single_turn_test.jsonl
│   │   ├── multi_turn_test.jsonl
│   │   └── domain_diverse_test.jsonl
│   └── reports/                       # Comparison reports
│       ├── phase1_comparison.json
│       └── csv/
│           ├── single_turn.csv
│           ├── json_quality.csv
│           └── llm_comparison.csv
│
├── run_phase1_experiment.sh           # NEW: Automated Phase 1 runner
├── RUN_EXPERIMENTS.md                 # NEW: Step-by-step guide
├── IMPLEMENTATION_SUMMARY.md          # NEW: This file
└── FUNCTION_CALLING_FINETUNE.md       # Existing paper reference
```

---

## Success Criteria (Phase 1)

### Primary Goal
**Match or surpass Azure LLM performance within 10% delta**

### Capability-Specific Thresholds

| Capability | Metric | Threshold | Why |
|-----------|--------|-----------|-----|
| **Single-turn** | Exact match rate | ≥ 70% | Direct mappings should be highly accurate |
| **Single-turn** | Field F1 | ≥ 0.80 | Near-perfect field extraction |
| **Multi-turn** | Field F1 | ≥ 0.60 | Context harder, lower threshold acceptable |
| **JSON validity** | Valid rate | ≥ 90% | Must produce parseable outputs |
| **Domain diversity** | F1 std dev | < 0.15 | Consistent across domains |

### Decision Gates

**After Phase 1 (100 examples):**
- ✅ If ANY approach achieves ≥90% of Azure LLM → **PROCEED TO PHASE 2**
- ✅ If Phi vs Llama shows clear winner → **Focus on winner**
- ❌ If all below 90% → **Iterate** (more data, hyperparams, prompt engineering)

**After Phase 2 (1000 examples):**
- ✅ If ≤10% delta to Azure → **DEPLOY TO PRODUCTION**
- ❌ If gap remains → **Analyze failure modes**, consider hybrid approach

---

## How to Run Experiments

### Quick Start (Automated)

```bash
# 1. Ensure .env file exists with API keys
cat .env

# 2. Run Phase 1 experiment (fully automated)
./run_phase1_experiment.sh

# 3. Wait ~90 minutes for completion

# 4. Review results
python -m json.tool comparison_framework/reports/phase1_comparison.json
```

### Manual Step-by-Step

See `RUN_EXPERIMENTS.md` for detailed manual instructions.

---

## Expected Results

### Fledgling Baseline (Existing)
Based on current implementation:
- **Structured track:** 96.67% JSON valid, 30% exact match, 0.624 F1
- **Toolcall track:** 0% performance (needs investigation)

### Paper Approach Predictions
Based on literature and similar fine-tuning:
- **Phi-4-mini:** Expected 85-90% of Llama performance (smaller model)
- **Llama-3.1-8B:** Expected to match or exceed Fledgling
- **Hermes dataset:** Should improve function calling vs xLAM (specialized dataset)

### Key Questions to Answer
1. Does Hermes dataset outperform xLAM for function calling?
2. Is Llama-3.1-8B worth the 2x model size vs Phi-4-mini?
3. Can either approach match Azure GPT-4 performance?
4. Which approach has better domain generalization?

---

## Next Steps

### Immediate (This Session)
1. ✅ **DONE:** Review this summary
2. ⏭️ **RUN:** Execute `./run_phase1_experiment.sh`
3. ⏭️ **ANALYZE:** Review results in `comparison_framework/reports/`
4. ⏭️ **DECIDE:** Proceed to Phase 2 or iterate

### If Phase 1 Succeeds
1. Scale up dataset to 1000/200/200 (Phase 2)
2. Retrain best-performing model
3. Re-evaluate on full test suite
4. Merge adapter and push to HuggingFace
5. Integrate into Mastra production

### If Phase 1 Fails
1. Analyze failure modes (per-example details in `*_details.jsonl`)
2. Iterate on:
   - Prompt engineering (system prompts)
   - Hyperparameters (LoRA rank, learning rate)
   - Dataset quality (filter low-quality examples)
   - Training duration (increase epochs)
3. Re-run Phase 1 with improvements

---

## Cost Analysis

### Compute Costs (Local)
- **Phase 1:** ~1.5 GPU-hours × 4 GPUs = 6 GPU-hours
- **Phase 2:** ~15 GPU-hours × 4 GPUs = 60 GPU-hours
- **Amortized cost:** ~$0.50/hour × 66 = **$33 total**

### API Costs (Azure Baseline)
- **Evaluation calls:** ~150 calls × $0.03/1K tokens × 500 avg tokens = **$2.25**

### Potential Savings (If Deployed)
- **Current:** Azure API at $30/1M tokens
- **After SLM:** Local inference at ~$0.10/1M tokens (amortized GPU)
- **ROI:** Break-even after ~1M tokens (~$30 API savings)

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Hermes dataset download fails | Check HF token, request access, or use synthetic fallback |
| Out of memory during training | Reduce batch size (2 or 1), increase gradient accumulation |
| Poor multi-turn performance | Check if dataset has multi-turn examples, may need to skip |
| Docker GPU not detected | Install nvidia-container-toolkit, restart Docker |
| Container build fails | Check internet connection, disk space (need ~10GB) |

---

## Architecture Comparison

### Fledgling (Current Approach)

**Pros:**
- ✅ Azure LLM comparison built-in (ground truth)
- ✅ Decision gate prevents unnecessary fine-tuning
- ✅ Two-track approach (structured + toolcall)
- ✅ Langfuse observability throughout

**Cons:**
- ❌ Requires Azure API access (cost + dependency)
- ❌ Complex pipeline (more moving parts)
- ❌ xLAM dataset may not be function-calling-optimized

### Paper Approach (FUNCTION_CALLING_FINETUNE)

**Pros:**
- ✅ Official Unsloth Docker (dependency isolation)
- ✅ Hermes dataset specialized for function calling
- ✅ Simpler architecture (direct fine-tuning)
- ✅ Supports multiple base models (Phi + Llama)

**Cons:**
- ❌ No built-in LLM comparison (manual evaluation)
- ❌ Single-track approach (unified format)
- ❌ Less observability (no Langfuse by default)

---

## Conclusion

We have built a **production-ready comparison framework** that enables:

1. ✅ **Side-by-side comparison** of two SLM fine-tuning approaches
2. ✅ **Iterative experimentation** (start small, scale up if successful)
3. ✅ **Comprehensive evaluation** across 4 key capabilities
4. ✅ **Automated execution** (one-click Phase 1 runner)
5. ✅ **Clear decision gates** (objective success criteria)

**Status:** Ready to execute. Run `./run_phase1_experiment.sh` to begin.

**Estimated Time to Results:** 90 minutes

**Recommendation:** Run Phase 1 overnight or during low-usage hours to maximize GPU utilization.

---

**Good luck with your experiments! 🚀**

For questions or issues, refer to:
- `RUN_EXPERIMENTS.md` - Detailed step-by-step guide
- `docker/*/README.md` - Docker-specific usage
- GitHub Issues: https://github.com/anthropics/claude-code/issues (for Claude Code feedback)
