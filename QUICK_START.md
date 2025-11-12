# Quick Start Guide - Function Calling Comparison

## TL;DR

Compare two approaches for fine-tuning SLMs to match LLM function calling performance.

**Time:** 90 minutes | **Cost:** ~$35 | **Goal:** ≤10% delta to Azure LLM

---

## Prerequisites Checklist

```bash
# 1. Check GPU access
nvidia-smi  # Should show 4x RTX 3090 (or similar)

# 2. Check Docker + NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. Create .env file
cat > .env << 'EOF'
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
HUGGING_FACE_HUB_TOKEN=hf_...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
EOF

# 4. Request Llama access (if needed)
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

---

## One-Command Run

```bash
./run_phase1_experiment.sh
```

**What it does:**
1. Downloads Hermes dataset (100 train, 50 val, 50 test)
2. Creates specialized test splits (single-turn, multi-turn, domain-diverse)
3. Builds Docker containers
4. Trains Phi-4-mini + Llama-3.1-8B in parallel
5. Evaluates both on all test suites
6. Generates comparison report

**When it's done:**
```bash
python -m json.tool comparison_framework/reports/phase1_comparison.json
```

---

## Manual Run (Step-by-Step)

### 1. Prepare Dataset (5 min)

```bash
python paper_approach/prepare_hermes_dataset.py \
  --train-size 100 --val-size 50 --test-size 50 --stratify
```

### 2. Create Test Splits (2 min)

```bash
# Single-turn
python comparison_framework/test_single_turn.py \
  --dataset-path paper_approach/datasets/hermes_test.jsonl \
  --output-path comparison_framework/test_splits/single_turn_test.jsonl

# Domain diversity
python comparison_framework/test_domain_diversity.py \
  --dataset-path paper_approach/datasets/hermes_test.jsonl \
  --output-path comparison_framework/test_splits/domain_diverse_test.jsonl
```

### 3. Start Docker Containers (2 min)

```bash
cd docker/unsloth-paper
docker-compose up -d
```

### 4. Train Models in Parallel (60-90 min)

```bash
# Terminal 1: Phi-4-mini (GPUs 0,1)
docker-compose exec unsloth-paper-phi bash
cd paper_approach/scripts
python train_function_call.py \
  --model-id microsoft/phi-4-mini \
  --dataset-path ../datasets/hermes_train.jsonl \
  --output-dir ../adapters/phi_hermes_lora

# Terminal 2: Llama-3.1-8B (GPUs 2,3)
docker-compose exec unsloth-paper-llama bash
cd paper_approach/scripts
python train_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path ../datasets/hermes_train.jsonl \
  --output-dir ../adapters/llama_hermes_lora
```

### 5. Evaluate (10 min)

```bash
# Phi-4-mini single-turn
docker-compose exec unsloth-paper-phi bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id microsoft/phi-4-mini \
  --adapter-path ../adapters/phi_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
  --output-path ../eval_results/phi_single_turn.json"

# Llama single-turn
docker-compose exec unsloth-paper-llama bash -c "cd paper_approach/scripts && \
python eval_function_call.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path ../adapters/llama_hermes_lora \
  --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
  --output-path ../eval_results/llama_single_turn.json"
```

### 6. Compare (1 min)

```bash
cd /home/gabriel/Desktop/AI_ATL25
python comparison_framework/compare_approaches.py
```

---

## Results Interpretation

### Success Criteria

| Metric | Target | Why |
|--------|--------|-----|
| JSON Valid Rate | ≥ 90% | Must produce parseable outputs |
| Exact Match (Single-turn) | ≥ 70% | Direct calls should be accurate |
| Field F1 (Single-turn) | ≥ 0.80 | Near-perfect field extraction |
| Delta to Azure LLM | ≤ 10% | Within acceptable performance gap |

### Decision Tree

```
Results < 90% of Azure
  ↓
  Iterate on Phase 1 (hyperparams, prompts, more data)

Results ≥ 90% of Azure
  ↓
  Proceed to Phase 2 (scale to 1000 examples)

Phase 2 Results ≤ 10% delta
  ↓
  Deploy to Production!
```

---

## Key Files to Check

```bash
# Dataset stats
cat paper_approach/datasets/dataset_stats.json

# Training metadata
cat paper_approach/adapters/phi_hermes_lora/metadata.json
cat paper_approach/adapters/llama_hermes_lora/metadata.json

# Evaluation results
cat paper_approach/eval_results/phi_single_turn.json | jq '.metrics'
cat paper_approach/eval_results/llama_single_turn.json | jq '.metrics'

# Comparison report
cat comparison_framework/reports/phase1_comparison.json | jq

# Detailed per-example results
head -n 10 paper_approach/eval_results/phi_single_turn_details.jsonl
```

---

## Troubleshooting

### Issue: "Hermes dataset not found"

```bash
# Check HF token
echo $HUGGING_FACE_HUB_TOKEN

# Request access
# Visit: https://huggingface.co/NousResearch/hermes-function-calling-v1

# Or use synthetic fallback (automatic)
```

### Issue: "Out of memory"

```bash
# Reduce batch size in training command
--batch-size 2  # or 1
--grad-accum 8  # increase to maintain effective batch size
```

### Issue: "GPU not detected in Docker"

```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "Container won't start"

```bash
# Check Docker logs
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## What to Expect

### Training Output (Example)

```
[1/3 epochs] Loss: 2.341 → 1.234
[2/3 epochs] Loss: 1.145 → 0.987
[3/3 epochs] Loss: 0.912 → 0.756
Validation: JSON valid 94.5%, Exact match 72.3%, F1 0.835
Saved to: paper_approach/adapters/phi_hermes_lora
```

### Evaluation Output (Example)

```json
{
  "metrics": {
    "valid_json_rate": 0.92,
    "name_match_rate": 0.88,
    "args_exact_match_rate": 0.74,
    "args_field_f1": 0.81
  }
}
```

### Comparison Output (Example)

```
COMPARISON REPORT
================================================================================
Single-Turn Performance:
  Phi-4-mini (Paper):  F1=0.81  (vs Azure: -0.08, within 10% ✓)
  Llama-3.1 (Paper):   F1=0.86  (vs Azure: -0.03, SURPASSES TARGET! ✓)
  Fledgling (Current): F1=0.62  (vs Azure: -0.27, FAILS ✗)

RECOMMENDATION: Deploy Llama-3.1 Paper approach to Phase 2
```

---

## Next Steps After Phase 1

### If Successful (≥90% of Azure)

```bash
# 1. Scale up dataset
python paper_approach/prepare_hermes_dataset.py \
  --train-size 1000 --val-size 200 --test-size 200

# 2. Retrain best model
# (Use same commands as Phase 1, but with new dataset)

# 3. Re-evaluate

# 4. If still successful, merge and deploy
```

### If Not Successful

```bash
# Analyze failure modes
head -n 50 paper_approach/eval_results/phi_single_turn_details.jsonl

# Common fixes:
# - Increase epochs (--epochs 5)
# - Increase LoRA rank (--lora-r 32)
# - Improve system prompts
# - Filter low-quality training examples
```

---

## Time Budget

| Phase | Activity | Time |
|-------|----------|------|
| Setup | Dataset + Docker build | 5-10 min |
| Training | Both models in parallel | 60-90 min |
| Evaluation | All test suites | 10-15 min |
| Analysis | Review reports | 5-10 min |
| **Total** | | **~90 min** |

---

## Resource Usage

| Resource | Phase 1 | Phase 2 |
|----------|---------|---------|
| GPU Hours | 6 hours | 60 hours |
| Disk Space | ~5 GB | ~20 GB |
| RAM | 32 GB | 64 GB |
| API Calls (Azure) | ~150 | ~1500 |
| Estimated Cost | $35 | $100 |

---

## Support

- **Detailed Guide:** `RUN_EXPERIMENTS.md`
- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Docker Guides:** `docker/*/README.md`
- **Fledgling Docs:** `slm_swap/README.md`

---

**Ready to start? Run:** `./run_phase1_experiment.sh`
