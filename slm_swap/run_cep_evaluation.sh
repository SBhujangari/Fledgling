#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="slm_swap/04_ft/adapter_llama_cep"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

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
echo "Adapter: $ADAPTER_PATH"
echo "Timestamp: $(date)"
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

echo ""

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
echo ""
echo "Toolcall Track:"
cat slm_swap/05_eval/toolcall_slm_cep_test.json
echo ""

# Compare against baselines
echo ""
echo "========================================"
echo "Comparison vs Azure LLM"
echo "========================================"
echo ""
echo "Structured Track:"
if python slm_swap/compare.py \
  --track structured \
  --azure slm_swap/05_eval/structured_azure_test.json \
  --slm slm_swap/05_eval/structured_slm_cep_test.json \
  --delta 0.0 2>/dev/null; then
  echo "✅ PASSED: CEP >= Azure baseline!"
else
  echo "❌ FAILED: CEP below Azure baseline"
fi

echo ""
echo "Toolcall Track:"
if python slm_swap/compare.py \
  --track toolcall \
  --azure slm_swap/05_eval/toolcall_azure_test.json \
  --slm slm_swap/05_eval/toolcall_slm_cep_test.json \
  --delta 0.0 2>/dev/null; then
  echo "✅ PASSED: CEP >= Azure baseline!"
else
  echo "❌ FAILED: CEP below Azure baseline"
fi

echo ""
echo "========================================"
echo "Improvement Analysis (Pre-CEP → CEP)"
echo "========================================"
echo ""

# Calculate deltas using Python
python3 - <<'PYEOF'
import json

# Load metrics
with open("slm_swap/05_eval/structured_slm_test.json") as f:
    pre_struct = json.load(f)
with open("slm_swap/05_eval/structured_slm_cep_test.json") as f:
    cep_struct = json.load(f)
with open("slm_swap/05_eval/toolcall_slm_test.json") as f:
    pre_tool = json.load(f)
with open("slm_swap/05_eval/toolcall_slm_cep_test.json") as f:
    cep_tool = json.load(f)

print("Structured Track:")
print(f"  Valid JSON: {pre_struct['json_valid_rate']:.1%} → {cep_struct['json_valid_rate']:.1%} ({cep_struct['json_valid_rate']-pre_struct['json_valid_rate']:+.1%})")
print(f"  Exact Match: {pre_struct['exact_match_rate']:.1%} → {cep_struct['exact_match_rate']:.1%} ({cep_struct['exact_match_rate']-pre_struct['exact_match_rate']:+.1%})")
print(f"  Field F1: {pre_struct['field_f1']:.1%} → {cep_struct['field_f1']:.1%} ({cep_struct['field_f1']-pre_struct['field_f1']:+.1%})")
print("")
print("Toolcall Track:")
print(f"  Valid Calls: {pre_tool['valid_call_rate']:.1%} → {cep_tool['valid_call_rate']:.1%} ({cep_tool['valid_call_rate']-pre_tool['valid_call_rate']:+.1%})")
print(f"  Name Match: {pre_tool['name_match_rate']:.1%} → {cep_tool['name_match_rate']:.1%} ({cep_tool['name_match_rate']-pre_tool['name_match_rate']:+.1%})")
print(f"  Args F1: {pre_tool['args_f1']:.1%} → {cep_tool['args_f1']:.1%} ({cep_tool['args_f1']-pre_tool['args_f1']:+.1%})")
PYEOF

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Full results:"
echo "  - slm_swap/05_eval/structured_slm_cep_test.json"
echo "  - slm_swap/05_eval/toolcall_slm_cep_test.json"
echo ""
echo "Detailed diagnostics:"
echo "  - slm_swap/05_eval/structured_slm_cep_test_details.jsonl"
echo "  - slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl"
echo ""
echo "Next: Review EVALUATION_PLAN.md for decision tree based on results"
echo ""
