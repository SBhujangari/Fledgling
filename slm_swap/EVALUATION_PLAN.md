# CEP Model Evaluation Plan

## Current Baseline Performance

### Structured Track (JSON Outputs)
| Metric | Azure LLM | SLM (pre-CEP) | Gap | Target (CEP) |
|--------|-----------|---------------|-----|--------------|
| Valid JSON Rate | 100% | 88% | -12% | >90% |
| Exact Match Rate | 20.5% | 4% | -16.5% | >20% |
| Field F1 | **60.2%** | 32.2% | **-28%** | **>60%** |

**Gap Analysis:** SLM is 87% worse than Azure (32.2% vs 60.2% F1)

### Toolcall Track (XML Tool Calls)
| Metric | Azure LLM | SLM (pre-CEP) | Gap | Target (CEP) |
|--------|-----------|---------------|-----|--------------|
| Valid Call Rate | 80% | 0% | -80% | >75% |
| Name Match Rate | 80% | 0% | -80% | >75% |
| Args F1 | **74.4%** | 0% | **-74.4%** | **>74%** |

**Gap Analysis:** SLM has complete failure mode (0% on all metrics)

---

## Why CEP Should Work

The pre-CEP SLM failures match **exactly** the error patterns CEP addresses:

### Observed Errors (from details):
1. ❌ **Markdown wrappers** - SLM outputs ```json blocks instead of raw JSON
2. ❌ **Self-closing XML tags** - `<tool_call name="func"/>` instead of `<tool_call>...</tool_call>`
3. ❌ **Parameter mismatches** - Wrong field names, missing required fields
4. ❌ **Invalid tool call format** - 100% of toolcall track fails format validation

### CEP Solution:
```
<|formatting_rules|>
CRITICAL OUTPUT RULES - FOLLOW EXACTLY:

1. JSON Outputs:
   - Output ONLY raw JSON object
   - NO markdown blocks (no ```json or ```)
   - Match parameter names EXACTLY as in signature

2. XML Tool Calls:
   - Format: <tool_call name="FUNC_NAME">{"arg": "value"}</tool_call>
   - MUST have opening tag: <tool_call name="...">
   - MUST have closing tag: </tool_call>
   - NO self-closing tags

VERIFY: Does output match format rules? Check before responding.
<|formatting_rules|>
```

**Expected Improvement:**
- Structured: 32% → **60%+** F1 (eliminate markdown wrappers, fix field matching)
- Toolcall: 0% → **74%+** F1 (fix XML format, enable valid parsing)

---

## Evaluation Workflow (Post-Training)

### Step 1: Wait for Training Completion
```bash
# Monitor training progress
tail -f slm_swap/logs/train_llama_cep_pure.log

# Check when complete (21/21 steps)
# Adapter saved to: slm_swap/04_ft/adapter_llama_cep/
```

**Current Status:** Step 4/21 (~19% complete, ~8-9 minutes remaining)

### Step 2: Run CEP Model Evaluation (Both Tracks)

#### 2a. Structured Track
```bash
python slm_swap/eval.py \
  --track structured \
  --model-kind slm \
  --split test \
  --adapter slm_swap/04_ft/adapter_llama_cep \
  --out slm_swap/05_eval/structured_slm_cep_test.json \
  --details-out slm_swap/05_eval/structured_slm_cep_test_details.jsonl \
  --batch-size 8
```

**Expected Output:** `slm_swap/05_eval/structured_slm_cep_test.json`
- Target: `field_f1 >= 0.60` (match or beat Azure)

#### 2b. Toolcall Track
```bash
python slm_swap/eval.py \
  --track toolcall \
  --model-kind slm \
  --split test \
  --adapter slm_swap/04_ft/adapter_llama_cep \
  --out slm_swap/05_eval/toolcall_slm_cep_test.json \
  --details-out slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl \
  --batch-size 8
```

**Expected Output:** `slm_swap/05_eval/toolcall_slm_cep_test.json`
- Target: `args_f1 >= 0.74` (match or beat Azure)

### Step 3: Compare Against Baselines

#### 3a. CEP vs Azure (Structured)
```bash
python slm_swap/compare.py \
  --track structured \
  --azure slm_swap/05_eval/structured_azure_test.json \
  --slm slm_swap/05_eval/structured_slm_cep_test.json \
  --delta 0.0
```

**Success Criteria:** Exit code 0 (delta <= 0.0, meaning CEP >= Azure)

#### 3b. CEP vs Azure (Toolcall)
```bash
python slm_swap/compare.py \
  --track toolcall \
  --azure slm_swap/05_eval/toolcall_azure_test.json \
  --slm slm_swap/05_eval/toolcall_slm_cep_test.json \
  --delta 0.0
```

**Success Criteria:** Exit code 0 (delta <= 0.0, meaning CEP >= Azure)

### Step 4: Generate Comparison Report
```bash
python slm_swap/generate_cep_report.py \
  --baseline-structured slm_swap/05_eval/structured_slm_test.json \
  --baseline-toolcall slm_swap/05_eval/toolcall_slm_test.json \
  --cep-structured slm_swap/05_eval/structured_slm_cep_test.json \
  --cep-toolcall slm_swap/05_eval/toolcall_slm_cep_test.json \
  --azure-structured slm_swap/05_eval/structured_azure_test.json \
  --azure-toolcall slm_swap/05_eval/toolcall_azure_test.json \
  --out slm_swap/reports/cep_evaluation_report.json
```

**Report Contents:**
- Side-by-side metrics comparison
- Delta calculations (pre-CEP → post-CEP)
- Success/failure analysis
- Next iteration recommendations

---

## Decision Tree (Based on Results)

### Scenario 1: CEP Beats LLM Baseline ✅
**Structured F1 >= 60% AND Toolcall F1 >= 74%**

**Next Steps:**
1. ✅ Mark CEP as successful
2. Train Phi-4 with CEP (smaller, faster model)
3. Integrate into production workflow tracer
4. Deploy gradual LLM → SLM swap
5. Monitor continuous improvement loop

**Commands:**
```bash
# Train Phi-4 with CEP
python slm_swap/train_phi_cep_v2.py \
  --model-id microsoft/Phi-4 \
  --train paper_approach/datasets/hermes_train.jsonl \
  --val paper_approach/datasets/hermes_val.jsonl \
  --output slm_swap/04_ft/adapter_phi_cep \
  --cep-type compact

# Evaluate Phi-4 CEP
# (same eval commands as above, with phi adapter path)
```

### Scenario 2: CEP Partially Succeeds 🟡
**One track succeeds, one fails**

**Example:** Structured >= 60% but Toolcall < 74%

**Next Steps:**
1. Analyze failure mode in details.jsonl
2. Refine CEP for failing track (track-specific variant)
3. Retrain with refined CEP
4. Re-evaluate

**Commands:**
```bash
# Analyze failures
grep '"issues":' slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl | head -20

# Common issues to look for:
# - "invalid_tool_call_format" → CEP not applied correctly
# - "tool_name_mismatch" → Wrong tool selected
# - "arguments_mismatch" → Field-level errors

# Refine CEP based on errors (manual editing)
vim slm_swap/cep_config.py

# Retrain with refined CEP
./slm_swap/resume_llama_cep.sh  # (with updated CEP)
```

### Scenario 3: CEP Fails Both Tracks ❌
**Structured < 60% AND Toolcall < 74%**

**Troubleshooting Steps:**
1. ✅ Verify CEP was actually applied during training
   - Check training logs for CEP prefix in system prompts
2. ✅ Verify model loaded adapter correctly
   - Check eval logs for adapter path
3. ✅ Analyze error patterns in details
   - Are errors the same as pre-CEP? (CEP not working)
   - Are errors different? (CEP introduced new issues)
4. ✅ Try alternative approaches:
   - Increase training data (100 → 1000 examples)
   - Increase epochs (3 → 5)
   - Try different base model (Llama → Mistral)
   - Try inference-time CEP only (no training)

**Commands:**
```bash
# Check if CEP applied
grep "formatting_rules" slm_swap/logs/train_llama_cep_pure.log

# Compare error patterns
diff \
  <(grep '"issues":' slm_swap/05_eval/toolcall_slm_test_details.jsonl | sort | uniq -c) \
  <(grep '"issues":' slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl | sort | uniq -c)

# Try inference-time CEP (no retraining needed)
python slm_swap/eval_with_inference_cep.py \
  --track toolcall \
  --adapter slm_swap/04_ft/adapter_llama_base \
  --cep-type universal
```

### Scenario 4: CEP Exceeds Expectations 🚀
**Structured F1 > 70% AND Toolcall F1 > 80%**

**Next Steps:**
1. 🎉 Celebrate! CEP is working better than expected
2. Scale up training (1000+ examples)
3. Push to HuggingFace Hub
4. Integrate into production immediately
5. Write research paper / blog post

**Commands:**
```bash
# Scale up training
python paper_approach/prepare_hermes_dataset.py \
  --train-size 1000 \
  --val-size 200 \
  --test-size 200

# Retrain with more data
python slm_swap/train_llama_cep_pure.py \
  --model-id unsloth/llama-3.1-8b-instruct-bnb-4bit \
  --train paper_approach/datasets/hermes_train.jsonl \
  --val paper_approach/datasets/hermes_val.jsonl \
  --output slm_swap/04_ft/adapter_llama_cep_v2 \
  --cep-type universal \
  --epochs 5

# Push to HuggingFace
python slm_swap/push_to_hf.py \
  --adapter slm_swap/04_ft/adapter_llama_cep_v2 \
  --repo-name "your-org/llama-3.1-8b-cep-function-calling" \
  --track both
```

---

## Automated Evaluation Script

Create `slm_swap/run_cep_evaluation.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="slm_swap/04_ft/adapter_llama_cep"

# Check adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
  echo "ERROR: Adapter not found at $ADAPTER_PATH"
  echo "Training may not be complete yet."
  exit 1
fi

echo "========================================"
echo "CEP Model Evaluation"
echo "========================================"
echo ""

# Structured track
echo "[1/2] Evaluating Structured Track..."
python slm_swap/eval.py \
  --track structured \
  --model-kind slm \
  --split test \
  --adapter "$ADAPTER_PATH" \
  --out slm_swap/05_eval/structured_slm_cep_test.json \
  --details-out slm_swap/05_eval/structured_slm_cep_test_details.jsonl \
  --batch-size 8

# Toolcall track
echo "[2/2] Evaluating Toolcall Track..."
python slm_swap/eval.py \
  --track toolcall \
  --model-kind slm \
  --split test \
  --adapter "$ADAPTER_PATH" \
  --out slm_swap/05_eval/toolcall_slm_cep_test.json \
  --details-out slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl \
  --batch-size 8

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""
echo "Structured Track:"
cat slm_swap/05_eval/structured_slm_cep_test.json
echo ""
echo "Toolcall Track:"
cat slm_swap/05_eval/toolcall_slm_cep_test.json
echo ""

# Compare against baselines
echo "========================================"
echo "Comparison vs Azure LLM"
echo "========================================"
echo ""
echo "Structured Track:"
python slm_swap/compare.py \
  --track structured \
  --azure slm_swap/05_eval/structured_azure_test.json \
  --slm slm_swap/05_eval/structured_slm_cep_test.json \
  --delta 0.0 || echo "❌ Below Azure baseline"

echo ""
echo "Toolcall Track:"
python slm_swap/compare.py \
  --track toolcall \
  --azure slm_swap/05_eval/toolcall_azure_test.json \
  --slm slm_swap/05_eval/toolcall_slm_cep_test.json \
  --delta 0.0 || echo "❌ Below Azure baseline"

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
```

---

## Timeline Estimates

| Task | Duration | Notes |
|------|----------|-------|
| Wait for training | ~8-9 min | Currently at step 4/21 |
| Structured eval | ~2-3 min | 50 examples × 8 batch size |
| Toolcall eval | ~2-3 min | 50 examples × 8 batch size |
| Analysis | ~5 min | Manual review of results |
| **Total** | **~20 min** | From now until decision |

---

## Success Metrics Summary

### Primary Goal
**Beat LLM baseline on both tracks**

| Track | Metric | Current SLM | Azure LLM | CEP Target | Status |
|-------|--------|-------------|-----------|------------|--------|
| Structured | Field F1 | 32.2% | **60.2%** | >=60% | ⏳ Pending |
| Toolcall | Args F1 | 0% | **74.4%** | >=74% | ⏳ Pending |

### Secondary Goals
- Valid JSON rate >= 90% (structured)
- Valid call rate >= 75% (toolcall)
- Exact match rate improvement (any increase is good)

---

## Next Actions (In Order)

1. ⏳ **Wait for training** (~8 min remaining)
   - Monitor: `tail -f slm_swap/logs/train_llama_cep_pure.log`

2. ✅ **Run evaluation script**
   - `./slm_swap/run_cep_evaluation.sh`

3. 📊 **Analyze results**
   - Review metrics vs targets
   - Examine error patterns in details

4. 🎯 **Make decision**
   - Follow decision tree above
   - Document findings
   - Plan next iteration

5. 🚀 **Execute next steps**
   - Based on decision tree outcome
   - Either: deploy, iterate, or troubleshoot

---

**Status:** Ready to execute once training completes!
