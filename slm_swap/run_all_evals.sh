#!/bin/bash
# Comprehensive evaluation runner for SLM Swap platform
# This runs all baseline evaluations before the critical LLM vs SLM comparison

set -e  # Exit on error

echo "========================================="
echo "SLM SWAP COMPREHENSIVE EVALUATION SUITE"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p slm_swap/05_eval

echo -e "${BLUE}Step 1: Generate 100-example evaluation datasets${NC}"
cd slm_swap
python create_100_eval.py
cd ..
echo -e "${GREEN}✓ Datasets created${NC}"
echo ""

echo -e "${BLUE}Step 2: Run baseline evaluations on existing test sets (50 examples each)${NC}"

echo -e "${YELLOW}2a. Evaluating SLM on structured track (test set)${NC}"
cd slm_swap
python eval.py --track structured --model-kind slm --split test
echo -e "${GREEN}✓ SLM structured test complete${NC}"
echo ""

echo -e "${YELLOW}2b. Evaluating SLM on toolcall track (test set)${NC}"
python eval.py --track toolcall --model-kind slm --split test
echo -e "${GREEN}✓ SLM toolcall test complete${NC}"
echo ""

echo -e "${YELLOW}2c. Evaluating Azure LLM on structured track (test set)${NC}"
python eval.py --track structured --model-kind azure --split test
echo -e "${GREEN}✓ Azure LLM structured test complete${NC}"
echo ""

echo -e "${YELLOW}2d. Evaluating Azure LLM on toolcall track (test set)${NC}"
python eval.py --track toolcall --model-kind azure --split test
echo -e "${GREEN}✓ Azure LLM toolcall test complete${NC}"
echo ""

echo -e "${BLUE}Step 3: Run extended 100-example evaluations${NC}"

echo -e "${YELLOW}3a. Evaluating SLM on structured track (100 examples)${NC}"
python eval.py --track structured --model-kind slm --split eval100
echo -e "${GREEN}✓ SLM structured eval100 complete${NC}"
echo ""

echo -e "${YELLOW}3b. Evaluating SLM on toolcall track (100 examples)${NC}"
python eval.py --track toolcall --model-kind slm --split eval100
echo -e "${GREEN}✓ SLM toolcall eval100 complete${NC}"
echo ""

echo -e "${YELLOW}3c. Evaluating Azure LLM on structured track (100 examples)${NC}"
python eval.py --track structured --model-kind azure --split eval100
echo -e "${GREEN}✓ Azure LLM structured eval100 complete${NC}"
echo ""

echo -e "${YELLOW}3d. Evaluating Azure LLM on toolcall track (100 examples)${NC}"
python eval.py --track toolcall --model-kind azure --split eval100
echo -e "${GREEN}✓ Azure LLM toolcall eval100 complete${NC}"
echo ""

echo -e "${RED}=========================================${NC}"
echo -e "${RED}CRITICAL: LLM vs SLM COMPARISON${NC}"
echo -e "${RED}=========================================${NC}"
echo ""

echo -e "${YELLOW}4a. Comparing LLM vs SLM on structured track (100 examples)${NC}"
python compare_llm_slm.py --track structured --split eval100
echo -e "${GREEN}✓ Structured comparison complete${NC}"
echo ""

echo -e "${YELLOW}4b. Comparing LLM vs SLM on toolcall track (100 examples)${NC}"
python compare_llm_slm.py --track toolcall --split eval100
echo -e "${GREEN}✓ Toolcall comparison complete${NC}"
echo ""

cd ..

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}ALL EVALUATIONS COMPLETE${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Results saved to slm_swap/05_eval/"
echo ""
echo "Check comparison results to determine if fine-tuning is needed:"
echo "  - slm_swap/05_eval/comparison_structured_eval100.json"
echo "  - slm_swap/05_eval/comparison_toolcall_eval100.json"
echo ""
echo "If performance gap > 15%, fine-tuning is recommended."
