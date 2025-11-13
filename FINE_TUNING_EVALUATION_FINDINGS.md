# Fine-Tuning Evaluation Findings Report

**Date:** 2025-11-12
**Session:** Hermes/CEP Evaluation Attempt
**Status:** CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

### Results Overview

Out of 3 fine-tuning attempts, only **1 succeeded**:

| Adapter | Status | Accuracy | vs Baseline | Root Cause |
|---------|--------|----------|-------------|------------|
| **Structured** | ✅ SUCCESS | 40.0% | +95% (vs 20.5%) | Correct training method |
| **Hermes** | ❌ FAILED | 0.0% | -100% (vs 64%) | Broken training script |
| **CEP** | ❌ CRASHED | N/A | N/A | Training error |

### Critical Finding

**The paper's training script (`docker/unsloth-paper/scripts/train_function_call.py`) is fundamentally broken and should NOT be used.**

---

## Detailed Results

### 1. Structured Format Adapter ✅

**Training:**
- Script: `slm_swap/train_llama_cep_pure.py` (CEP-based approach)
- Duration: ~5 minutes
- Loss: Started 0.5 → Final 0.58
- Method: Proper Llama 3.1 chat template tokens

**Evaluation:**
- Test set: 50 examples
- Exact match accuracy: **40.0%**
- Azure LLM baseline: 20.5%
- **Improvement: +95%**

**Why it worked:**
- Used actual Llama tokens: `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`
- Evaluation used `tokenizer.apply_chat_template()` matching training
- Format consistency throughout pipeline

**Evidence:**
```
File: eval_structured_results.json
{
  "total": 50,
  "correct": 20,
  "accuracy": 0.4
}
```

### 2. Hermes Format Adapter ❌

**Training:**
- Script: `docker/unsloth-paper/scripts/train_function_call.py`
- Duration: 9 minutes 19 seconds
- Loss: Started 0.86 → Final 0.76 (MISLEADING - loss decreased but task not learned)
- Method: Custom fake tokens

**Evaluation:**
- Test set: 50 examples
- Valid JSON rate: **0.0%**
- Name match rate: **0.0%**
- Args exact match: **0.0%**
- Azure baseline: 64% args exact, 74% F1
- **Complete failure**

**Why it failed:**
- Training used custom tokens: `<|system|>`, `<|user|>`, `<|assistant|>`
- These tokens are NOT recognized by Llama 3.1
- Model learned to generate exercise questions instead of tool calls
- Even with matching eval format: still 0% accuracy

**Example output:**
```
Prompt: "extract queries from the following passage..."
Expected: {"arguments": {"queries": [...]}, "name": "ExpertQAExtractor"}
Got: "Can you help me extract queries... #### EXERCISES..."
```

**Evidence:**
```
File: paper_approach/eval_results/llama_hermes_correct_format.json
{
  "valid_json_rate": 0.0,
  "name_match_rate": 0.0,
  "args_exact_match_rate": 0.0
}
```

### 3. CEP Format Adapter ❌

**Training:**
- Script: `slm_swap/train_llama_cep.py`
- Status: **CRASHED**
- Error: `AttributeError: 'int' object has no attribute 'mean'`
- Step: Failed at step 0/6

**Why it crashed:**
- Implementation bug in training loop
- Adapter files exist but training incomplete
- Cannot be evaluated

**Evidence:**
```
File: slm_swap/logs/train_llama_cep.log (lines 315-330)
Traceback (most recent call last):
  File "/home/gabriel/Desktop/AI_ATL25/slm_swap/train_llama_cep.py", line 179
    trainer_stats = trainer.train()
AttributeError: 'int' object has no attribute 'mean'
```

---

## Root Cause Analysis

### The Fundamental Problem

**Broken Training Script:** `docker/unsloth-paper/scripts/train_function_call.py`

**Lines 42-52 (THE PROBLEM):**
```python
def format_hermes_example(example):
    text = ""
    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            text += f"<|system|>\n{value}\n\n"      # ❌ WRONG
        elif from_role == "human":
            text += f"<|user|>\n{value}\n\n"        # ❌ WRONG
        elif from_role == "gpt":
            text += f"<|assistant|>\n{value}\n"     # ❌ WRONG

    return {"text": text}
```

**Why this is wrong:**
1. `<|system|>`, `<|user|>`, `<|assistant|>` are NOT Llama 3.1 tokens
2. These are custom placeholders with no meaning to the model
3. Llama 3.1 uses: `<|start_header_id|>role<|end_header_id|>` format
4. Training with fake tokens = model learns gibberish patterns

### The Correct Approach

**Working Script:** `slm_swap/train_llama_cep_pure.py`

**Lines 12-17 (CORRECT):**
```python
def format_and_tokenize(example):
    conversations = example["conversations"]
    text = ""

    for msg in conversations:
        from_role = msg.get("from", "")
        value = msg.get("value", "")

        if from_role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{value}<|eot_id|>"  # ✅ CORRECT
        elif from_role == "human":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{value}<|eot_id|>"                    # ✅ CORRECT
        elif from_role == "gpt":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{value}<|eot_id|>"               # ✅ CORRECT

    text += "<|end_of_text|>"
    return tokenizer(text, ...)
```

**Why this works:**
1. Uses actual Llama 3.1 special tokens
2. These tokens are defined in the tokenizer vocabulary
3. Model understands these tokens from pre-training
4. Training teaches task-specific behavior, not token recognition

---

## Technical Investigation Timeline

### Initial State (Start of Session)
- Structured adapter: Already working (40% accuracy)
- Hermes adapter: Trained, not yet evaluated
- CEP adapter: Trained, not yet evaluated

### Discovery 1: Chat Template Issue (First Evaluation Attempt)
- Ran Hermes evaluation with original script
- Got 0% on all metrics after 22 minutes
- Model generated conversational text instead of tool calls
- **Hypothesis:** Evaluation not using chat templates

### Discovery 2: Chat Template "Fix" (Actually Made it Worse)
- Modified `eval_function_call.py` to use `tokenizer.apply_chat_template()`
- Re-ran evaluation
- Still got 0% on all metrics
- Model now generated template echoes instead of tool calls
- **Hypothesis was wrong:** Chat templates weren't the issue

### Discovery 3: Format Mismatch (Root Cause Identified)
- Compared training scripts:
  - Hermes: Used `<|system|>` custom tokens
  - CEP/Structured: Used `<|start_header_id|>` real Llama tokens
- **Key insight:** Training and eval BOTH need to match, AND use correct tokens

### Discovery 4: Revert and Retry (Confirmed Failure)
- Reverted eval script to use custom tokens (matching training)
- Re-ran evaluation for 20 minutes
- Still got 0% on all metrics
- **Conclusion:** Training itself was broken, not just eval

### Discovery 5: Training Script is Fundamentally Broken
- Model generates exercise questions instead of tool calls
- Loss decreased during training (0.86 → 0.76) but didn't learn task
- **Final conclusion:** Custom token approach completely fails

---

## Why Decreasing Loss Doesn't Mean Success

### What Happened with Hermes

**Training metrics looked good:**
```
Epoch 1: loss=0.86
Epoch 2: loss=0.76
Eval loss: 0.76
```

**But evaluation showed:**
```
Valid JSON: 0%
Name match: 0%
Args exact: 0%
```

### Explanation

The model DID learn something during training:
- It learned the statistical patterns of the training data
- It learned to predict the next token in the custom format
- Loss decreased because predictions improved

But it learned the WRONG thing:
- It didn't learn to call functions
- It learned to generate text that looks like the training format
- When given a real task, it generates unrelated content

**Analogy:**
Teaching someone to respond to `<|system|>` instead of "System says:" is like teaching them a language that doesn't exist. They can get good at mimicking the fake language (decreasing loss) but can't communicate in it (0% task performance).

---

## Key Lessons Learned

### 1. Use the Model's Actual Tokens
**❌ DON'T:** Create custom tokens like `<|system|>`, `<|user|>`
**✅ DO:** Use the tokenizer's defined special tokens

```python
# Check what tokens the model actually knows:
tokenizer.special_tokens_map
# Returns: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>', ...}
```

### 2. Format Consistency is Necessary but Not Sufficient
**Not enough:** Training format = Evaluation format
**Required:** BOTH use correct format that model understands

### 3. Decreasing Loss ≠ Learning the Task
- Loss measures prediction accuracy on training data
- Task performance measures actual capability
- Always evaluate on held-out test set

### 4. Debugging Requires Systematic Investigation
Our process:
1. Initial eval → 0% (identified problem exists)
2. Check model output → gibberish (identified format issue)
3. Compare training scripts → found token mismatch
4. Re-eval with matching format → still 0% (confirmed training broken)
5. Inspect training script → found root cause

### 5. The Paper's Approach Was Not Validated
- `docker/unsloth-paper/` scripts appear to be experimental/WIP
- No validation that they actually work
- Should not be used for production training

---

## Context Engineering Fairness (Separate Issue)

**Note:** This is DIFFERENT from the format issue.

From `CONTEXT_ENGINEERING_ANALYSIS.md`:
- **Structured format:** LLM baseline has system prompts, SLM training doesn't (unfair)
- **Hermes/CEP formats:** Both have system prompts (fair, but training broken anyway)

**To-do:** Once Hermes is retrained correctly, structured format still needs system prompts added to training data for fair comparison.

---

## File References

### Scripts

**Working:**
- `slm_swap/train_llama_cep_pure.py` - CEP training (CORRECT approach)
- `eval_structured_quick.py` - Structured evaluation (works with chat templates)

**Broken:**
- `docker/unsloth-paper/scripts/train_function_call.py` - Hermes training (DO NOT USE)
- `docker/unsloth-paper/scripts/eval_function_call.py` - Initially broken, fixed, then reverted

### Data

**Training:**
- `slm_swap/02_dataset/structured/train_chat.jsonl` - Structured training data
- `paper_approach/datasets/hermes_train.jsonl` - Hermes training data (300 examples)
- `paper_approach/datasets/hermes_test.jsonl` - Hermes test data (50 examples)

**Adapters:**
- `slm_swap/04_ft/adapter_llama_structured/` - WORKS (321MB)
- `slm_swap/04_ft/adapter_llama_hermes/` - BROKEN (321MB, but learned wrong thing)
- `slm_swap/04_ft/adapter_llama_cep/` - INCOMPLETE (321MB, training crashed)

### Results

**Evaluation outputs:**
- `eval_structured_results.json` - 40% accuracy ✅
- `paper_approach/eval_results/llama_hermes_correct_format.json` - 0% accuracy ❌
- `paper_approach/eval_results/llama_hermes_correct_format_details.jsonl` - Detailed failures

**Baselines:**
- `slm_swap/05_eval/structured_azure_test.json` - Azure LLM: 20.5% exact match
- `slm_swap/05_eval/toolcall_azure_test.json` - Azure LLM: 64% args exact, 74% F1

### Logs

- `slm_swap/logs/train_llama_hermes.log` - Hermes training (loss: 0.86→0.76)
- `slm_swap/logs/train_llama_cep.log` - CEP training crash

---

## Recommendations

### Immediate Actions

1. **DO NOT use `docker/unsloth-paper/scripts/train_function_call.py`**
   - This script is fundamentally broken
   - Will waste GPU time and produce useless adapters

2. **Use CEP-based training for all formats**
   - Copy `slm_swap/train_llama_cep_pure.py` approach
   - Adapt for Hermes dataset
   - Ensure proper Llama tokens used

3. **Fix CEP training crash**
   - Debug the AttributeError in `slm_swap/train_llama_cep.py`
   - Likely issue with return type in training loop

### Retrain Hermes Correctly

**New training script needed:** `slm_swap/train_llama_hermes_fixed.py`

```python
# Pseudocode for correct Hermes training:
def format_hermes_correct(example):
    conversations = example["conversations"]
    text = ""

    for msg in conversations:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "human":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "gpt":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

    text += "<|end_of_text|>"

    # Tokenize using the tokenizer (not manual text assembly)
    return tokenizer(text, truncation=True, max_length=2048)
```

**Evaluation script:** Use `tokenizer.apply_chat_template()` for consistency

**Expected result:** Should match or exceed structured format's 40% accuracy

### Production Checklist

Before deploying any fine-tuned adapter:

- [ ] Training script uses `tokenizer.special_tokens_map` tokens
- [ ] Evaluation uses `tokenizer.apply_chat_template()`
- [ ] Format identical between training and eval
- [ ] Test set evaluation shows >0% accuracy
- [ ] Manual inspection of outputs shows task understanding
- [ ] Baseline comparison proves fine-tuning value
- [ ] Context engineering is equivalent between SLM and LLM baselines

---

## Success Criteria for Future Work

### Hermes Format (Redo)
- [ ] Retrain with correct Llama tokens
- [ ] Achieve >50% args exact match (vs 64% Azure baseline)
- [ ] Achieve >60% F1 score (vs 74% Azure baseline)
- [ ] Manual inspection confirms tool calls generated correctly

### CEP Format (Fix and Train)
- [ ] Fix AttributeError in training script
- [ ] Successfully complete 3 epochs
- [ ] Achieve >50% args exact match
- [ ] Compare performance vs standard Hermes approach

### Structured Format (Fair Comparison)
- [ ] Add system prompts to training data
- [ ] Retrain with prompt parity
- [ ] Re-evaluate with fair comparison
- [ ] Likely result: >40% (current) with fair prompting

---

## Conclusion

**What we proved:**
- Fine-tuned SLMs CAN outperform LLMs on specialized tasks (40% vs 20.5%)
- Proper training methodology is CRITICAL
- Format consistency alone is insufficient

**What we learned:**
- Custom tokens don't work - use model's actual tokens
- Decreasing loss doesn't guarantee task learning
- Systematic debugging is essential
- The paper's scripts need complete rewrite

**Next steps:**
1. Fix Hermes training script (use Llama tokens)
2. Fix CEP training crash
3. Retrain both adapters
4. Add system prompts to structured training for fair comparison
5. Comprehensive evaluation against Azure baselines

**Status:**
- **Structured adapter:** Production-ready (with caveat about unfair prompting)
- **Hermes adapter:** Must be retrained from scratch
- **CEP adapter:** Must be debugged and trained from scratch

---

## Related Documents

- `CONTEXT_ENGINEERING_ANALYSIS.md` - Prompt fairness analysis
- `SLM_VS_LLM_RESULTS.md` - Structured format success story
- `IMPLEMENTATION_SUMMARY.md` - Project overview
- `FORMAT_ANALYSIS_REPORT.md` - LLM-as-judge format comparison

**This document should be read FIRST before attempting any fine-tuning work.**
