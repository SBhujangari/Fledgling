#!/bin/bash
# Quick script to run just the LLM vs SLM comparison
# Use this if you've already run baseline evals and just need the comparison

set -e

echo "========================================="
echo "LLM vs SLM COMPARISON (FINE-TUNE DECISION)"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

cd slm_swap

# Check if datasets exist
if [ ! -f "02_dataset/structured/eval100.jsonl" ] || [ ! -f "02_dataset/toolcall/eval100.jsonl" ]; then
    echo -e "${YELLOW}Creating 100-example datasets...${NC}"
    python create_100_eval.py
    echo -e "${GREEN}✓ Datasets created${NC}"
    echo ""
fi

echo -e "${RED}Running LLM vs SLM comparisons...${NC}"
echo ""

echo -e "${YELLOW}1. Structured track comparison (100 examples)${NC}"
python compare_llm_slm.py --track structured --split eval100

echo ""
echo -e "${YELLOW}2. Toolcall track comparison (100 examples)${NC}"
python compare_llm_slm.py --track toolcall --split eval100

cd ..

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}COMPARISON COMPLETE${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - slm_swap/05_eval/comparison_structured_eval100.json"
echo "  - slm_swap/05_eval/comparison_toolcall_eval100.json"
echo ""
echo "Review the 'needs_fine_tuning' and 'recommendation' fields in the results."
