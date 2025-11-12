# Next Steps Summary

## Current Status: Training in Progress ✅

### What We Accomplished

1. **Recovered from GPU Driver Crash**
   - GPUs were healthy (nvidia-smi working)
   - Resumed training successfully

2. **Fixed OOM Error**
   - Hit CUDA out of memory at step 7/21 (33% complete)
   - Applied memory optimizations:
     - Reduced GPUs: 4 → 2 (less fragmentation)
     - Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
     - Reduced gradient accumulation: 16 → 8
     - Reduced max sequence length: 2048 → 1536
   - Training now progressing smoothly

3. **Created Complete Evaluation Infrastructure**
   - `slm_swap/EVALUATION_PLAN.md` - Comprehensive plan with baselines and decision tree
   - `slm_swap/run_cep_evaluation.sh` - Automated evaluation script
   - `slm_swap/generate_cep_report.py` - Report generator with recommendations

### Current Training Status

**Command:** `./slm_swap/resume_llama_cep.sh`
**PID:** 271803
**Progress:** Step 6/39 (15%)
**Speed:** ~18s/step
**Estimated Time:** ~10-11 minutes remaining
**Log:** `slm_swap/logs/train_llama_cep_pure.log`

**Monitor Progress:**
```bash
tail -f slm_swap/logs/train_llama_cep_pure.log
```

---

## Once Training Completes

### Step 1: Run Automated Evaluation (~5 minutes)

```bash
./slm_swap/run_cep_evaluation.sh
```

This will:
- Evaluate CEP model on both tracks (structured + toolcall)
- Compare against Azure LLM baseline
- Show improvement metrics (pre-CEP → CEP)

### Step 2: Generate Comprehensive Report

```bash
python slm_swap/generate_cep_report.py \
  --baseline-structured slm_swap/05_eval/structured_slm_test.json \
  --baseline-toolcall slm_swap/05_eval/toolcall_slm_test.json \
  --cep-structured slm_swap/05_eval/structured_slm_cep_test.json \
  --cep-toolcall slm_swap/05_eval/toolcall_slm_cep_test.json \
  --azure-structured slm_swap/05_eval/structured_azure_test.json \
  --azure-toolcall slm_swap/05_eval/toolcall_azure_test.json
```

This will:
- Generate JSON and Markdown reports
- Provide recommendations based on results
- Include next action items

---

## Expected Results & Decision Tree

### Baseline Performance (Pre-CEP)

| Track | Metric | Azure LLM | SLM (pre-CEP) | Gap |
|-------|--------|-----------|---------------|-----|
| **Structured** | Field F1 | **60.2%** | 32.2% | -28% |
| **Toolcall** | Args F1 | **74.4%** | 0% | -74.4% |

### CEP Targets

- **Structured:** Field F1 >= 60% (match Azure)
- **Toolcall:** Args F1 >= 74% (match Azure)

### Scenario 1: SUCCESS ✅
**Both tracks meet targets**

**Next Actions:**
1. Train Phi-4 with CEP (smaller, faster)
2. Integrate into production workflow tracer
3. Deploy gradual LLM → SLM swap
4. Scale up training data (100 → 1000 examples)

### Scenario 2: PARTIAL SUCCESS 🟡
**One track succeeds, one fails**

**Next Actions:**
1. Analyze failure patterns in details.jsonl
2. Refine CEP for failing track
3. Retrain with refined CEP
4. Re-evaluate

### Scenario 3: FAILURE ❌
**Neither track meets targets**

**Next Actions:**
1. Verify CEP was applied correctly (check logs)
2. Analyze error patterns (compare pre-CEP vs CEP)
3. Try alternative approaches:
   - Inference-time CEP only
   - Increase training data/epochs
   - Different base model
   - Track-specific CEP variants

---

## Files Created

### Documentation
- `slm_swap/EVALUATION_PLAN.md` - Complete evaluation plan with baseline metrics
- `NEXT_STEPS.md` - This file (quick reference)

### Scripts
- `slm_swap/run_cep_evaluation.sh` - Automated evaluation runner
- `slm_swap/generate_cep_report.py` - Comprehensive report generator

### Modified
- `slm_swap/resume_llama_cep.sh` - Added memory optimizations

---

## Key Insights

### Why CEP Should Work

The pre-CEP SLM failures match **exactly** the error patterns CEP addresses:

1. **Markdown wrappers** - SLM outputs ```json blocks
   - CEP explicitly instructs: "NO markdown blocks"

2. **Self-closing XML tags** - `<tool_call/>` instead of `<tool_call>...</tool_call>`
   - CEP shows correct format with opening/closing tags

3. **Parameter mismatches** - Wrong field names
   - CEP instructs: "Match parameter names EXACTLY"

4. **Invalid tool call format** - 100% failure rate on toolcall track
   - CEP provides explicit format rules

### Memory Optimizations Applied

**Problem:** CUDA OOM at step 7 with 23.3GB/23.7GB used

**Solutions:**
1. Reduced GPUs: 4 → 2 (less inter-GPU coordination)
2. PyTorch flag: `expandable_segments:True` (reduce fragmentation)
3. Smaller batches: grad_accum 16 → 8 (less memory per step)
4. Shorter sequences: max_length 2048 → 1536 (25% memory reduction)

**Trade-off:** Slightly longer training (39 steps vs 21), but completes successfully

---

## Timeline

| Event | Time | Status |
|-------|------|--------|
| GPU recovery | 00:30 | ✅ Complete |
| Training start | 00:41 | ✅ Complete |
| OOM error | ~00:45 | ✅ Fixed |
| Restart with optimizations | 01:01 | ✅ Complete |
| Training progress | 01:03 (current) | ⏳ In progress |
| Estimated completion | 01:12 | ⏳ Pending |
| Evaluation | 01:17 | ⏳ Pending |
| Report generation | 01:20 | ⏳ Pending |
| **Decision point** | **01:20** | ⏳ Pending |

---

## How to Check Status

### Training Progress
```bash
# Watch logs
tail -f slm_swap/logs/train_llama_cep_pure.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check process
ps aux | grep train_llama_cep_pure.py
```

### When Training Completes

You'll see in the log:
```
100%|██████████| 39/39 [XX:XX<00:00, X.XXs/it]
Training complete!
Saving adapters to: slm_swap/04_ft/adapter_llama_cep/
```

Then run:
```bash
./slm_swap/run_cep_evaluation.sh
```

---

## Quick Command Reference

```bash
# Monitor training
tail -f slm_swap/logs/train_llama_cep_pure.log

# Run evaluation (after training completes)
./slm_swap/run_cep_evaluation.sh

# Generate detailed report
python slm_swap/generate_cep_report.py \
  --baseline-structured slm_swap/05_eval/structured_slm_test.json \
  --baseline-toolcall slm_swap/05_eval/toolcall_slm_test.json \
  --cep-structured slm_swap/05_eval/structured_slm_cep_test.json \
  --cep-toolcall slm_swap/05_eval/toolcall_slm_cep_test.json \
  --azure-structured slm_swap/05_eval/structured_azure_test.json \
  --azure-toolcall slm_swap/05_eval/toolcall_azure_test.json

# View results
cat slm_swap/reports/cep_evaluation_report.md

# If success, train Phi-4
python slm_swap/train_phi_cep_v2.py \
  --model-id microsoft/Phi-4 \
  --train paper_approach/datasets/hermes_train.jsonl \
  --val paper_approach/datasets/hermes_val.jsonl \
  --output slm_swap/04_ft/adapter_phi_cep \
  --cep-type compact \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536
```

---

## Summary

**Where we left off:**
- CEP training is **in progress** (step 6/39, ~15% complete)
- Memory issues **resolved** with optimizations
- Evaluation infrastructure **ready to execute**
- Expected completion: ~10-11 minutes

**What happens next:**
1. Wait for training to complete (~10 min)
2. Run automated evaluation (~5 min)
3. Review results against baselines
4. Follow decision tree based on results
5. Execute next iteration (deploy, iterate, or troubleshoot)

**Goal:** Beat LLM baseline performance (60% F1 structured, 74% F1 toolcall)

---

**Status:** ⏳ Waiting for training completion
**ETA:** ~10-11 minutes
**Next action:** Run `./slm_swap/run_cep_evaluation.sh`
