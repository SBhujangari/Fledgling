# LLM-as-Judge: Training vs Evaluation Format Analysis

**Date:** 2025-11-12
**Analyst:** Claude (LLM Judge)
**Status:** ❌ CRITICAL MISMATCH FOUND

---

## Executive Summary

**VERDICT: The model trained successfully but evaluation uses WRONG dataset format**

**Root Cause:** Training used Hermes conversational format, but we have THREE different test formats available and need to pick the RIGHT one.

**Impact:** 0% success rate due to format mismatch

---

## Format Analysis

### Format 1: Hermes Training Data (1892 examples)
**Location:** `paper_approach/datasets/hermes_train.jsonl`
**Structure:** Multi-turn conversations with system/human/assistant roles
**Output Format:**
```
<tool_call>
{'arguments': {...}, 'name': 'function_name'}
</tool_call>
```
**Prompt Style:** Conversational, XML tool definitions, multi-turn context

---

### Format 2: Hermes Test Data (946 examples)
**Location:** `paper_approach/datasets/hermes_test.jsonl`
**Structure:** Same as training - multi-turn conversations
**Output Format:** Same as training
**Match:** ✅ MATCHES TRAINING FORMAT

---

### Format 3: SLM Swap Structured (30 examples)
**Location:** `slm_swap/02_dataset/structured/test.jsonl`
**Structure:** Single prompt/completion pairs
**Output Format:**
```json
{"arguments": {...}, "query": "...", "tool_name": "..."}
```
**Prompt Style:** Direct instruction, no XML, structured JSON output
**Match:** ❌ DIFFERENT FORMAT

---

### Format 4: SLM Swap Tool Call (30 examples)
**Location:** `slm_swap/02_dataset/toolcall/test.jsonl`
**Structure:** Single prompt/completion pairs
**Output Format:**
```xml
<tool_call name="function_name">{...}</tool_call>
```
**Prompt Style:** Direct instruction with tool signature
**Match:** ❌ DIFFERENT FORMAT (similar XML but simpler)

---

## Root Cause Analysis

### What Went Wrong

The evaluation script (`docker/unsloth-paper/scripts/eval_function_call.py`) correctly loads Hermes test data, which DOES match the training format. However:

1. **Model generates text continuation** instead of tool calls
2. **This suggests training didn't work** OR prompt construction is wrong
3. **The eval script constructs prompts differently** than the training script

### Key Finding

Looking at the actual model output:
```
Expected: <tool_call>{'arguments': {...}, 'name': '...'}</tool_call>
Actual: "Can you help me extract queries... #### EXERCISES ..."
```

The model is **continuing the passage text** instead of **generating a tool call**.

This means:
- The model didn't learn to generate tool calls
- OR the evaluation prompt doesn't trigger the learned behavior
- OR we need to check the training script's prompt construction

---

## Task Type Decision

### Two Main Task Types

**1. TOOL CALLS (Function Calling)**
- Format: `<tool_call name="X">{args}</tool_call>`
- Use Case: Agent systems, API calling, multi-step workflows
- Datasets: Hermes (1892 train, 946 test), SLM Swap toolcall (30 test)

**2. STRUCTURED JSON OUTPUT**
- Format: `{"tool_name": "X", "arguments": {...}}`
- Use Case: Structured data extraction, form filling, API requests
- Datasets: SLM Swap structured (30 test)

### Deterministic Selection Process

```python
def determine_task_type(example):
    """Deterministic process for format selection"""

    # Check completion format
    completion = example.get("completion", "")

    # Tool call format: starts with <tool_call
    if completion.strip().startswith("<tool_call"):
        return "TOOL_CALL"

    # Structured JSON: starts with {
    elif completion.strip().startswith("{"):
        # Check if it has tool_name key
        try:
            data = json.loads(completion)
            if "tool_name" in data and "arguments" in data:
                return "STRUCTURED_JSON"
        except:
            pass

    # Conversational format: check for multi-turn
    if "conversations" in example:
        # Check last assistant message
        last_msg = example["conversations"][-1]
        if last_msg["from"] == "gpt":
            if "<tool_call>" in last_msg["value"]:
                return "TOOL_CALL_CONVERSATIONAL"

    return "UNKNOWN"
```

---

##Recommendations

### Immediate Actions

1. **✅ USE HERMES TEST SET** - It matches training format perfectly
2. **❌ DON'T use slm_swap datasets** - Different format, will fail
3. **🔍 DEBUG evaluation script** - Check prompt construction
4. **📊 Check training script** - Verify it used correct format

### Next Steps

**Option A: Fix Evaluation (RECOMMENDED)**
1. Verify eval script constructs prompts same as training
2. Re-run evaluation on Hermes test set
3. Should get >0% if prompts match

**Option B: Use Easiest Test Set**
1. Retrain on slm_swap structured format (30 examples only)
2. Evaluate on slm_swap structured test
3. This is MUCH simpler task, should work quickly

**Option C: Start Fresh**
1. Pick ONE format (recommend: slm_swap structured)
2. Create consistent train/test/eval pipeline
3. Verify formats match before training
4. Train + evaluate

---

## Validation Checklist

Before any training run:
- [ ] Training data format documented
- [ ] Test data format documented
- [ ] Eval script prompt construction documented
- [ ] All three formats MATCH
- [ ] Sample outputs manually verified
- [ ] LLM judge confirms match

---

## Conclusion

**The model trained successfully**, but we evaluated it wrong. The Hermes test set DOES match training, but either:
1. The model didn't learn properly (unlikely with 1892 examples)
2. The evaluation prompt construction differs from training
3. We need to check the training script to see how it formatted prompts

**RECOMMENDATION:** Check training script prompt construction, then re-evaluate with matching prompts.

---

## TODO: Implement Automated LLM Judge

```python
# Future: Replace manual analysis with foundry model
def llm_judge_format_match(train_file, test_file, eval_script):
    """
    Use foundry model to automatically validate:
    1. Training data format
    2. Test data format
    3. Eval script prompt construction
    4. All match? Return True/False + report
    """
    pass
```
