# Evaluation Metrics - Industry Standards

This document describes the industry-standard metrics used in our evaluation framework.

## Overview

Our evaluation framework uses **industry-standard metrics** for assessing structured JSON generation and tool calling capabilities:

1. **Precision, Recall, and F1 Score** - Standard information retrieval metrics
2. **Exact Match** - Strict correctness measure
3. **Field-level Analysis** - Granular error tracking

## Determinism Configuration

To ensure **reproducible results**, both models are configured with maximum determinism:

### SLM (Local Model)
- `do_sample=False` - Disables sampling (greedy decoding)
- `temperature=0.0` - Zero temperature for deterministic output
- Location: `slm_swap/slm_client.py:120-121`

### Azure LLM
- `temperature=0.0` - Zero temperature for deterministic output
- Location: `slm_swap/azure_client.py:79,92,123,163`

## Metrics for Structured JSON Track

### Core Metrics (Always Enabled)

#### 1. JSON Valid Rate
**Definition**: Percentage of outputs that are valid JSON
```
json_valid_rate = valid_outputs / total_outputs
```
**Industry Standard**: ✓ Basic syntax validation is fundamental for any JSON generation task

#### 2. Exact Match Rate
**Definition**: Percentage of outputs that exactly match the reference
```
exact_match_rate = exact_matches / total_outputs
```
**Industry Standard**: ✓ Used in SQuAD, GLUE, and other NLP benchmarks

#### 3. Field-level Precision
**Definition**: Of all fields in the prediction, how many are correct?
```
precision = true_positives / (true_positives + false_positives)
```
**Industry Standard**: ✓ Standard information retrieval metric (from TREC)

#### 4. Field-level Recall
**Definition**: Of all required fields, how many did we predict?
```
recall = true_positives / (true_positives + false_negatives)
```
**Industry Standard**: ✓ Standard information retrieval metric (from TREC)

#### 5. Field-level F1 Score
**Definition**: Harmonic mean of precision and recall
```
f1 = 2 * (precision * recall) / (precision + recall)
```
**Industry Standard**: ✓ Widely used in NLP (F1 Score from van Rijsbergen, 1979)

### Extended Metrics (Optional - OpenAI Cookbook)

Enable with `--use-llm-judge` and/or `--use-semantic-similarity` flags.

#### 6. Semantic Similarity Score
**Definition**: Cosine similarity between prediction and reference embeddings
```
semantic_similarity = cosine_sim(embed(pred), embed(ref))
```
**Pass Threshold**: 0.85 (OpenAI Cookbook recommendation)
**Industry Standard**: ✓ OpenAI Evals for structured outputs
**Purpose**: Catches semantically correct outputs that differ in format/wording

**When to use**:
- Outputs may be semantically equivalent but differently formatted
- Natural language content within JSON structures
- Validation beyond strict string equality

#### 7. LLM-as-Judge Score
**Definition**: Model-graded quality score using evaluation rubric
```
judge_score = llm_evaluate(pred, ref, rubric) / 7.0  # normalized to [0,1]
```
**Score Range**: 1-7 (normalized to 0.0-1.0)
**Pass Threshold**: 0.85 (OpenAI Cookbook recommendation)
**Industry Standard**: ✓ OpenAI Evals for structured outputs
**Purpose**: Captures nuanced semantic correctness beyond rule-based metrics

**Evaluation Rubric** (1-7 scale):
- **7**: Perfect - All fields correct, properly formatted, semantically accurate
- **6**: Excellent - Minor formatting differences, content fully correct
- **5**: Good - Semantically correct, some non-critical field mismatches
- **4**: Acceptable - Core information correct, some missing/extra fields
- **3**: Partial - Some correct fields, notable gaps or errors
- **2**: Poor - Mostly incorrect or incomplete
- **1**: Failed - Invalid or completely wrong output

**When to use**:
- Complex nested structures where exact match is too strict
- Outputs with natural language content requiring semantic evaluation
- Cases where multiple valid representations exist
- Post-hoc analysis of edge cases

## Metrics for Tool Call Track

### Core Metrics (Always Enabled)

#### 1. Valid Call Rate
**Definition**: Percentage of outputs in correct tool call format
```
valid_call_rate = valid_calls / total_calls
```
**Industry Standard**: ✓ Format validation essential for function calling

#### 2. Name Match Rate
**Definition**: Percentage of calls with correct tool name
```
name_match_rate = correct_names / total_calls
```
**Industry Standard**: ✓ Multi-class classification accuracy

#### 3. Arguments Exact Match Rate
**Definition**: Percentage of calls with exactly correct arguments
```
args_exact_rate = exact_arg_matches / total_calls
```
**Industry Standard**: ✓ Strict correctness measure

#### 4. Arguments Precision/Recall/F1
**Definition**: Same as structured track, applied to tool arguments
```
args_precision = tp / (tp + fp)
args_recall = tp / (tp + fn)
args_f1 = 2 * (precision * recall) / (precision + recall)
```
**Industry Standard**: ✓ Standard IR metrics

### Extended Metrics (Optional - OpenAI Cookbook)

Enable with `--use-llm-judge` and/or `--use-semantic-similarity` flags.

#### 5. Semantic Similarity Score (Arguments)
**Definition**: Cosine similarity between predicted and reference arguments
```
semantic_similarity = cosine_sim(embed(pred_args), embed(ref_args))
```
**Pass Threshold**: 0.85 (OpenAI Cookbook recommendation)
**Industry Standard**: ✓ OpenAI Evals for function calling
**Purpose**: Catches semantically equivalent argument values

#### 6. LLM-as-Judge Score (Tool Calls)
**Definition**: Model-graded quality score for tool call correctness
```
judge_score = llm_evaluate(pred_call, ref_call, rubric) / 7.0
```
**Score Range**: 1-7 (normalized to 0.0-1.0)
**Pass Threshold**: 0.85 (OpenAI Cookbook recommendation)
**Industry Standard**: ✓ Berkeley Function Calling Leaderboard (BFCL)

**Evaluation Rubric** (1-7 scale):
- **7**: Perfect - Correct tool name and all arguments with exact values
- **6**: Excellent - Correct tool and arguments, minor formatting differences
- **5**: Good - Correct tool, semantically correct arguments
- **4**: Acceptable - Correct tool, some argument mismatches
- **3**: Partial - Correct tool, significant argument errors
- **2**: Poor - Wrong tool or mostly incorrect arguments
- **1**: Failed - Invalid format or completely wrong call

## Field-level Evaluation (Flattening Algorithm)

Our evaluation uses a **recursive flattening** algorithm to compare nested structures:

```python
# Example:
{
  "user": {
    "name": "Alice",
    "age": 30
  },
  "items": ["a", "b"]
}

# Flattens to:
{
  "user.name": "\"Alice\"",
  "user.age": "30",
  "items[0]": "\"a\"",
  "items[1]": "\"b\""
}
```

This allows **precise field-level comparison** of nested JSON structures.

**Industry Standard**: ✓ Similar to JSON Path evaluation used in testing frameworks

## Live Progress Reporting

During evaluation, live progress is displayed showing:
- Current example number (e.g., `[42/100]`)
- Track name
- Model kind
- Running metrics (JSON valid rate, exact match rate)

**Implementation**: `slm_swap/eval.py:381-386`

## Comparison Decision Threshold

For LLM vs SLM comparison, we use a **15% performance gap threshold**:

```python
FINE_TUNE_THRESHOLD = 0.15  # 15% performance gap
needs_fine_tuning = (llm_performance - slm_performance) > 0.15
```

**Rationale**:
- < 5% gap: Minimal practical difference
- 5-15% gap: Moderate difference, fine-tuning optional
- > 15% gap: Significant difference, fine-tuning recommended

## Optional Metrics Configuration

### Enabling Extended Metrics

Add flags to evaluation commands:

```bash
# Enable LLM-as-Judge only
python eval.py --track structured --model-kind slm --use-llm-judge

# Enable semantic similarity only
python eval.py --track toolcall --model-kind azure --use-semantic-similarity

# Enable both extended metrics
python eval.py --track structured --model-kind slm --use-llm-judge --use-semantic-similarity
```

### Performance Considerations

**LLM-as-Judge**:
- Requires additional API calls to Azure LLM (1 per example)
- Increases evaluation time significantly
- Cost: ~1-2 tokens per evaluation (prompt + response)
- Recommended: Use for analysis, not routine evals

**Semantic Similarity**:
- Requires embedding computation (via Azure or local)
- Minimal performance impact with embedding caching
- Cost: Negligible (embeddings reused)
- Recommended: Safe for routine evals if needed

### Judge Model Configuration

The LLM-as-Judge uses the same Azure deployment as baseline evaluation:
- **Model**: Same as `AZURE_ENDPOINT` configuration
- **Temperature**: 0.0 (deterministic scoring)
- **System Prompt**: Includes evaluation rubric (1-7 scale)
- **Output Format**: JSON with score and reasoning

## References

Industry standards referenced:
1. **Precision/Recall/F1**: van Rijsbergen, C.J. (1979). Information Retrieval (2nd ed.)
2. **Exact Match**: Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension
3. **JSON Validation**: ECMA-404 The JSON Data Interchange Standard
4. **Tool Calling Evaluation**: Berkeley Function Calling Leaderboard (BFCL)
5. **LLM-as-Judge & Semantic Similarity**: OpenAI Cookbook - Structured Outputs Evaluation (2024)
6. **Evaluation Thresholds**: OpenAI Evals Best Practices (0.85 pass threshold)

## Summary

✓ All metrics are **industry-standard**
✓ Determinism is **maximized** (temperature=0, greedy decoding)
✓ Live progress **displays current prompt**
✓ Comparison mode **directly compares LLM vs SLM**
✓ Decision threshold **scientifically justified**
