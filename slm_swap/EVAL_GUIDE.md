# Evaluation Guide - Quick Start

## Overview

This guide shows you how to run comprehensive evaluations including the critical **LLM vs SLM comparison** that determines if fine-tuning is needed.

## Prerequisites

Make sure you have:
1. Azure OpenAI credentials configured (`.env` file)
2. Local SLM model downloaded (default: `models/phi-4-mini`)
3. Required Python packages installed

## Quick Start - Run Everything

To run all evaluations in one command:

```bash
./slm_swap/run_all_evals.sh
```

This will:
1. ✓ Generate 100-example evaluation datasets
2. ✓ Run baseline evals on test sets (50 examples)
3. ✓ Run extended evals on 100 examples
4. ✓ **Compare LLM vs SLM to determine if fine-tuning is needed**

## Individual Commands

### 1. Generate 100-Example Datasets

```bash
cd slm_swap
python create_100_eval.py
```

This creates:
- `02_dataset/structured/eval100.jsonl` (100 examples)
- `02_dataset/toolcall/eval100.jsonl` (100 examples)

### 2. Run Individual Evaluations

**Evaluate SLM on structured track:**
```bash
cd slm_swap
python eval.py --track structured --model-kind slm --split eval100
```

**Evaluate Azure LLM on structured track:**
```bash
cd slm_swap
python eval.py --track structured --model-kind azure --split eval100
```

**Evaluate SLM on toolcall track:**
```bash
cd slm_swap
python eval.py --track toolcall --model-kind slm --split eval100
```

**Evaluate Azure LLM on toolcall track:**
```bash
cd slm_swap
python eval.py --track toolcall --model-kind azure --split eval100
```

### 3. Run LLM vs SLM Comparison (CRITICAL)

**This is the main evaluation that decides if fine-tuning is needed:**

**Structured track comparison:**
```bash
cd slm_swap
python compare_llm_slm.py --track structured --split eval100
```

**Toolcall track comparison:**
```bash
cd slm_swap
python compare_llm_slm.py --track toolcall --split eval100
```

## Live Progress

All evaluations show live progress:
```
[42/100] track=structured model=slm json_valid_rate=95.24% exact_match_rate=85.71%
```

You can see:
- Current prompt number (42 out of 100)
- Track and model being evaluated
- Running metrics updated in real-time

## Understanding Results

### Individual Evaluation Results

Results are saved to `slm_swap/05_eval/`:
- `structured_slm_eval100.json` - SLM performance on structured tasks
- `structured_azure_eval100.json` - LLM performance on structured tasks
- `toolcall_slm_eval100.json` - SLM performance on tool calling
- `toolcall_azure_eval100.json` - LLM performance on tool calling

### Comparison Results (CRITICAL)

**Location:**
- `slm_swap/05_eval/comparison_structured_eval100.json`
- `slm_swap/05_eval/comparison_toolcall_eval100.json`

**Example output:**
```json
{
  "track": "structured",
  "split": "eval100",
  "total_examples": 100,
  "llm_metrics": {
    "json_valid_rate": 0.98,
    "exact_match_rate": 0.92,
    "field_f1": 0.95
  },
  "slm_metrics": {
    "json_valid_rate": 0.94,
    "exact_match_rate": 0.78,
    "field_f1": 0.82
  },
  "performance_gap": 0.14,
  "needs_fine_tuning": false,
  "recommendation": "SLM performance acceptable, fine-tuning optional"
}
```

### Decision Threshold

- **< 15% gap**: SLM is good enough, fine-tuning optional
- **> 15% gap**: Fine-tuning recommended (significant performance difference)

## Evaluation with Fine-Tuned Adapter

If you have a fine-tuned LoRA adapter:

```bash
cd slm_swap
python compare_llm_slm.py --track structured --split eval100 --adapter path/to/adapter
```

## Metrics Explained

All metrics are industry-standard (see `METRICS.md` for details):

- **Exact Match Rate**: Percentage of perfect predictions
- **JSON Valid Rate**: Percentage of syntactically valid outputs
- **Field Precision**: Of predicted fields, how many are correct?
- **Field Recall**: Of required fields, how many were predicted?
- **Field F1**: Harmonic mean of precision and recall

## Determinism Settings

Both models use maximum determinism for reproducible results:
- **Temperature**: 0.0 (both LLM and SLM)
- **Sampling**: Disabled (greedy decoding)

This ensures consistent results across runs.

## Troubleshooting

### "Dataset is empty" error
Make sure you've run `create_100_eval.py` first to generate the datasets.

### Azure content filter errors
Some prompts may be filtered by Azure's content policy. These are automatically skipped.

### CUDA out of memory
The SLM runs in 8-bit quantization. If you still get OOM errors, try:
- Reducing `max_new_tokens` in `slm_client.py`
- Using a smaller model
- Running on a machine with more GPU memory

## Next Steps

After running comparisons:

1. **If performance gap > 15%**: Run fine-tuning
   ```bash
   # See fine-tuning guide
   ```

2. **If performance gap < 15%**: SLM is production-ready
   - Deploy SLM for cost/latency benefits
   - Keep monitoring production metrics

## Summary Commands

**Run all baseline evaluations before comparison:**
```bash
./slm_swap/run_all_evals.sh
```

**Or run just the critical LLM vs SLM comparison:**
```bash
cd slm_swap
python compare_llm_slm.py --track structured --split eval100
python compare_llm_slm.py --track toolcall --split eval100
```
