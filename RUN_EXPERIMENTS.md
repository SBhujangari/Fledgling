# Step-by-Step Experimental Guide
## Comparing Fledgling vs Paper Approach for LLM → SLM Function Calling

This guide implements an **iterative experimental approach** to prove that SLM fine-tuning can match or surpass LLM performance across:
1. Single-turn function calling
2. Multi-turn conversations
3. Structured JSON outputs
4. Diverse domains

---

## Phase 0: Environment Setup

### Prerequisites

- 4x NVIDIA RTX 3090 GPUs (or similar)
- Docker with NVIDIA container toolkit
- Hugging Face account with access to:
  - `meta-llama/Llama-3.1-8B-Instruct` (gated - requires Meta approval)
  - Hermes Function Calling dataset
- Azure OpenAI API access (for Fledgling baseline)

### Setup Steps

```bash
# 1. Clone or navigate to project
cd /home/gabriel/Desktop/AI_ATL25

# 2. Create .env file
cat > .env << 'EOF'
# Azure OpenAI (for Fledgling baseline)
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Langfuse (optional but recommended)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Hugging Face (required for Llama and Hermes dataset)
HUGGING_FACE_HUB_TOKEN=hf_...
EOF

# 3. Build Docker images
cd docker/fledgling && docker-compose build
cd ../unsloth-paper && docker-compose build
cd ../..

# 4. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Phase 1: Dataset Preparation (Iterative Start - Small Scale)

### Step 1.1: Download and Prepare Hermes Dataset

```bash
# Start with small splits for rapid iteration
python paper_approach/prepare_hermes_dataset.py \
  --dataset-name NousResearch/hermes-function-calling-v1 \
  --output-dir paper_approach/datasets \
  --train-size 100 \
  --val-size 50 \
  --test-size 50 \
  --stratify \
  --seed 42
```

**Expected Output:**
- `paper_approach/datasets/hermes_train.jsonl` (100 examples)
- `paper_approach/datasets/hermes_val.jsonl` (50 examples)
- `paper_approach/datasets/hermes_test.jsonl` (50 examples)
- `paper_approach/datasets/dataset_stats.json`

### Step 1.2: Create Specialized Test Splits

```bash
# Single-turn test split (50 examples: 20 simple + 20 moderate + 10 complex)
python comparison_framework/test_single_turn.py \
  --dataset-path paper_approach/datasets/hermes_test.jsonl \
  --output-path comparison_framework/test_splits/single_turn_test.jsonl \
  --simple 20 --moderate 20 --complex 10

# Multi-turn test split (30 examples, if available)
python comparison_framework/test_multi_turn.py \
  --dataset-path paper_approach/datasets/hermes_test.jsonl \
  --output-path comparison_framework/test_splits/multi_turn_test.jsonl \
  --clarification 10 --sequential 10 --context 10

# Domain diversity test split (50 examples: 5 per domain)
python comparison_framework/test_domain_diversity.py \
  --dataset-path paper_approach/datasets/hermes_test.jsonl \
  --output-path comparison_framework/test_splits/domain_diverse_test.jsonl \
  --per-domain 5
```

---

## Phase 2: Baseline Evaluation (Fledgling Current Approach)

### Step 2.1: Run Fledgling Baseline

```bash
# Start Fledgling container
cd docker/fledgling
docker-compose up -d
docker-compose exec fledgling bash

# Inside container:
cd slm_swap

# Evaluate Azure baseline (structured track)
python eval.py \
  --track structured \
  --model-kind azure \
  --test-path 02_dataset/structured/test.jsonl \
  --output-path 05_eval/structured_azure_test.json

# Evaluate SLM baseline (structured track)
python eval.py \
  --track structured \
  --model-kind slm \
  --test-path 02_dataset/structured/test.jsonl \
  --output-path 05_eval/structured_slm_test.json \
  --batch-size 8

# Evaluate Azure baseline (toolcall track)
python eval.py \
  --track toolcall \
  --model-kind azure \
  --test-path 02_dataset/toolcall/test.jsonl \
  --output-path 05_eval/toolcall_azure_test.json

# Evaluate SLM baseline (toolcall track)
python eval.py \
  --track toolcall \
  --model-kind slm \
  --test-path 02_dataset/toolcall/test.jsonl \
  --output-path 05_eval/toolcall_slm_test.json \
  --batch-size 8

# Compare and decide if fine-tuning is needed
bash run_comparison.sh
```

**Expected Metrics:**
- Azure baseline: High performance (target to match)
- SLM baseline: Initial performance (likely low)
- Decision gate: If delta > 10%, trigger fine-tuning

### Step 2.2: Fine-tune Fledgling (If Triggered)

```bash
# If comparison shows gap > 10%, run fine-tuning
bash train_parallel.sh  # 2+2 GPU allocation

# Or for maximum per-model resources:
bash train_sequential.sh  # 4 GPUs per track

# After training, re-evaluate
python eval.py \
  --track structured \
  --model-kind slm \
  --adapter 04_ft/adapter_structured \
  --test-path 02_dataset/structured/test.jsonl \
  --output-path 05_eval/structured_slm_ft_test.json \
  --batch-size 8
```

---

## Phase 3: Paper Approach Training (Both Models in Parallel)

### Step 3.1: Train Phi-4-mini (Paper Approach)

```bash
# Start paper approach containers
cd docker/unsloth-paper
docker-compose up -d

# Train Phi-4-mini on GPUs 0,1
docker-compose exec unsloth-paper-phi bash
cd paper_approach/scripts

python train_function_call.py \
  --model-id microsoft/phi-4-mini \
  --dataset-path ../datasets/hermes_train.jsonl \
  --val-dataset-path ../datasets/hermes_val.jsonl \
  --output-dir ../adapters/phi_hermes_lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --epochs 3 \
  --max-seq-length 2048
```

**Expected Training Time:** ~30-45 minutes on 2x RTX 3090

### Step 3.2: Train Llama-3.1-8B (Paper Approach)

```bash
# In parallel, train Llama on GPUs 2,3
# Open new terminal
docker-compose exec unsloth-paper-llama bash
cd paper_approach/scripts

python train_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path ../datasets/hermes_train.jsonl \
  --val-dataset-path ../datasets/hermes_val.jsonl \
  --output-dir ../adapters/llama_hermes_lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --epochs 3 \
  --max-seq-length 2048
```

**Expected Training Time:** ~60-90 minutes on 2x RTX 3090

---

## Phase 4: Iterative Evaluation (Prove Capabilities)

### Step 4.1: Single-Turn Function Calling

```bash
# Evaluate Phi-4-mini (Paper)
docker-compose exec unsloth-paper-phi bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id microsoft/phi-4-mini \
  --adapter-path ../adapters/phi_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
  --output-path ../eval_results/phi_single_turn.json"

# Evaluate Llama-3.1-8B (Paper)
docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path ../adapters/llama_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
  --output-path ../eval_results/llama_single_turn.json"
```

**Success Criteria (Phase 1 - Small Scale):**
- ✅ Valid JSON rate ≥ 90%
- ✅ Exact match rate ≥ 70% (single-turn)
- ✅ Field F1 ≥ 0.80

### Step 4.2: Multi-Turn Conversations

```bash
# Evaluate on multi-turn split
docker-compose exec unsloth-paper-phi bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id microsoft/phi-4-mini \
  --adapter-path ../adapters/phi_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/multi_turn_test.jsonl \
  --output-path ../eval_results/phi_multi_turn.json"

docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path ../adapters/llama_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/multi_turn_test.jsonl \
  --output-path ../eval_results/llama_multi_turn.json"
```

**Success Criteria:**
- ✅ Field F1 ≥ 0.60 (multi-turn is harder)
- ✅ Name match rate ≥ 80%

### Step 4.3: Structured JSON Output

```bash
# Evaluate JSON validity and structure
# This is already captured in valid_json_rate from previous evals
# Check detailed results:

cat paper_approach/eval_results/phi_single_turn.json | jq '.metrics'
cat paper_approach/eval_results/llama_single_turn.json | jq '.metrics'
```

**Success Criteria:**
- ✅ JSON valid rate ≥ 90%
- ✅ Field precision ≥ 0.85
- ✅ Field recall ≥ 0.85

### Step 4.4: Domain Diversity

```bash
# Evaluate on domain-diverse split
docker-compose exec unsloth-paper-phi bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id microsoft/phi-4-mini \
  --adapter-path ../adapters/phi_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/domain_diverse_test.jsonl \
  --output-path ../eval_results/phi_domain_diverse.json"

docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path ../adapters/llama_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/domain_diverse_test.jsonl \
  --output-path ../eval_results/llama_domain_diverse.json"
```

**Success Criteria:**
- ✅ Consistent F1 across domains (std dev < 0.15)
- ✅ No domain with F1 < 0.50

---

## Phase 5: Comprehensive Comparison

### Step 5.1: Generate Comparison Report

```bash
# Run comparison framework
python comparison_framework/compare_approaches.py \
  --fledgling-results slm_swap/05_eval \
  --paper-results paper_approach/eval_results \
  --output comparison_framework/reports/phase1_comparison.json
```

**Expected Output:**
- Comparison tables for all 4 capabilities
- CSV exports for easy analysis
- Decision on whether to proceed to Phase 2 (scale up)

### Step 5.2: Analyze Results

```bash
# View comparison results
cat comparison_framework/reports/phase1_comparison.json | jq

# Check LLM comparison
cat comparison_framework/reports/csv/llm_comparison.csv
```

**Decision Gate:**
- If **any approach** achieves ≥90% of Azure LLM performance → **PROCEED TO PHASE 2**
- If Phi-4-mini vs Llama shows clear winner → **Focus resources on winner**
- If all below 90% → **Iterate on training (more data, hyperparams)**

---

## Phase 2: Scale Up (If Phase 1 Succeeds)

### Step 2.1: Increase Dataset Size

```bash
# Prepare larger dataset (1000 train, 200 val, 200 test)
python paper_approach/prepare_hermes_dataset.py \
  --dataset-name NousResearch/hermes-function-calling-v1 \
  --output-dir paper_approach/datasets_large \
  --train-size 1000 \
  --val-size 200 \
  --test-size 200 \
  --stratify \
  --seed 42
```

### Step 2.2: Retrain with Larger Dataset

```bash
# Retrain best-performing model from Phase 1
# Example: If Llama outperformed Phi

docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && \
python train_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path ../datasets_large/hermes_train.jsonl \
  --val-dataset-path ../datasets_large/hermes_val.jsonl \
  --output-dir ../adapters/llama_hermes_lora_large \
  --epochs 5"  # Increase epochs for larger dataset
```

### Step 2.3: Re-evaluate on Full Test Suite

```bash
# Re-run all 4 evaluation suites
# Single-turn, multi-turn, JSON, domain diversity
```

---

## Phase 3: Production Deployment (If Phase 2 Succeeds)

### Merge and Export Adapter

```bash
# Merge LoRA adapter into base model
python docker/unsloth-paper/scripts/merge_adapter.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path paper_approach/adapters/llama_hermes_lora_large \
  --output-path paper_approach/merged_models/llama_funcall_merged

# Push to Hugging Face
python slm_swap/hf_upload.py \
  paper_approach/merged_models/llama_funcall_merged \
  --repo-id your-username/llama-funcall-8b-merged \
  --private
```

---

## Summary of Success Criteria

| Phase | Capability | Metric | Threshold |
|-------|-----------|--------|-----------|
| 1 | Single-turn | Exact match | ≥ 70% |
| 1 | Single-turn | Field F1 | ≥ 0.80 |
| 1 | Multi-turn | Field F1 | ≥ 0.60 |
| 1 | JSON validity | Valid rate | ≥ 90% |
| 1 | Domain diversity | Std dev F1 | < 0.15 |
| 1 | **LLM comparison** | **Delta to Azure** | **≤ 10%** |
| 2 | All above | Same thresholds | On 10x dataset |
| 3 | Production | Match/surpass LLM | ≤ 0% delta |

---

## Troubleshooting

### Issue: Hermes dataset download fails

**Solution:**
```bash
# Check HF token
echo $HUGGING_FACE_HUB_TOKEN

# Request access at https://huggingface.co/NousResearch/hermes-function-calling-v1

# Or use fallback synthetic examples (automatically generated)
```

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size
--batch-size 2

# Increase gradient accumulation
--grad-accum 8

# Reduce LoRA rank
--lora-r 8
```

### Issue: Poor performance on multi-turn

**Solution:**
- Check if dataset has multi-turn examples
- If not, focus on single-turn and domain diversity
- Consider creating synthetic multi-turn examples

---

## Next Steps After Success

1. **Integrate into Mastra**: Replace Azure OpenAI calls with local SLM
2. **Benchmark latency**: Compare inference speed (SLM should be faster)
3. **A/B testing**: Run parallel Azure + SLM, compare quality in production
4. **Cost analysis**: Calculate savings (Azure API vs GPU amortization)
5. **Continuous improvement**: Add new examples from production errors

---

**Good luck with your experiments! 🚀**
