#!/bin/bash
# Automated Phase 1 Experiment Runner
# Runs small-scale iterative comparison between Fledgling and Paper approaches

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/gabriel/Desktop/AI_ATL25"

echo "========================================================================"
echo "Phase 1 Experiment: Fledgling vs Paper Approach Comparison"
echo "Small-scale iterative validation"
echo "========================================================================"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create .env with required API keys (see RUN_EXPERIMENTS.md)"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not installed!${NC}"
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA Docker runtime not available!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Step 1: Prepare Hermes dataset
echo -e "${YELLOW}[2/7] Preparing Hermes dataset (100 train, 50 val, 50 test)...${NC}"

if [ ! -d "$PROJECT_ROOT/paper_approach/datasets" ]; then
    python3 "$PROJECT_ROOT/paper_approach/prepare_hermes_dataset.py" \
        --dataset-name NousResearch/hermes-function-calling-v1 \
        --output-dir "$PROJECT_ROOT/paper_approach/datasets" \
        --train-size 100 \
        --val-size 50 \
        --test-size 50 \
        --stratify \
        --seed 42

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset prepared${NC}"
    else
        echo -e "${YELLOW}⚠ Dataset download failed, using synthetic examples${NC}"
    fi
else
    echo -e "${GREEN}✓ Dataset already exists${NC}"
fi
echo ""

# Step 2: Create test splits
echo -e "${YELLOW}[3/7] Creating specialized test splits...${NC}"

mkdir -p "$PROJECT_ROOT/comparison_framework/test_splits"

# Single-turn
python3 "$PROJECT_ROOT/comparison_framework/test_single_turn.py" \
    --dataset-path "$PROJECT_ROOT/paper_approach/datasets/hermes_test.jsonl" \
    --output-path "$PROJECT_ROOT/comparison_framework/test_splits/single_turn_test.jsonl" \
    --simple 20 --moderate 20 --complex 10

# Multi-turn (may have no examples)
python3 "$PROJECT_ROOT/comparison_framework/test_multi_turn.py" \
    --dataset-path "$PROJECT_ROOT/paper_approach/datasets/hermes_test.jsonl" \
    --output-path "$PROJECT_ROOT/comparison_framework/test_splits/multi_turn_test.jsonl" \
    --clarification 10 --sequential 10 --context 10 || echo "No multi-turn examples found"

# Domain diversity
python3 "$PROJECT_ROOT/comparison_framework/test_domain_diversity.py" \
    --dataset-path "$PROJECT_ROOT/paper_approach/datasets/hermes_test.jsonl" \
    --output-path "$PROJECT_ROOT/comparison_framework/test_splits/domain_diverse_test.jsonl" \
    --per-domain 5

echo -e "${GREEN}✓ Test splits created${NC}"
echo ""

# Step 3: Build Docker images
echo -e "${YELLOW}[4/7] Building Docker images...${NC}"

echo "Building Fledgling container..."
cd "$PROJECT_ROOT/docker/fledgling"
docker-compose build --quiet

echo "Building Paper approach containers..."
cd "$PROJECT_ROOT/docker/unsloth-paper"
docker-compose build --quiet

cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

# Step 4: Train Paper approach (both models in parallel)
echo -e "${YELLOW}[5/7] Training Paper approach models (this will take ~60-90 minutes)...${NC}"

cd "$PROJECT_ROOT/docker/unsloth-paper"
docker-compose up -d

echo "Training Phi-4-mini on GPUs 0,1..."
docker-compose exec -T unsloth-paper-phi bash -c "cd paper_approach/scripts && \
    python train_function_call.py \
        --model-id microsoft/phi-4-mini \
        --dataset-path ../datasets/hermes_train.jsonl \
        --val-dataset-path ../datasets/hermes_val.jsonl \
        --output-dir ../adapters/phi_hermes_lora \
        --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
        --batch-size 4 --grad-accum 4 --lr 2e-4 --epochs 3" &

PHI_PID=$!

echo "Training Llama-3.1-8B on GPUs 2,3..."
docker-compose exec -T unsloth-paper-llama bash -c "cd paper_approach/scripts && \
    python train_function_call.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --dataset-path ../datasets/hermes_train.jsonl \
        --val-dataset-path ../datasets/hermes_val.jsonl \
        --output-dir ../adapters/llama_hermes_lora \
        --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
        --batch-size 4 --grad-accum 4 --lr 2e-4 --epochs 3" &

LLAMA_PID=$!

# Wait for both trainings to complete
wait $PHI_PID
PHI_EXIT=$?

wait $LLAMA_PID
LLAMA_EXIT=$?

if [ $PHI_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Phi-4-mini training complete${NC}"
else
    echo -e "${RED}✗ Phi-4-mini training failed${NC}"
fi

if [ $LLAMA_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Llama-3.1-8B training complete${NC}"
else
    echo -e "${RED}✗ Llama-3.1-8B training failed${NC}"
fi

cd "$PROJECT_ROOT"
echo ""

# Step 5: Evaluate all models
echo -e "${YELLOW}[6/7] Evaluating all models on test splits...${NC}"

cd "$PROJECT_ROOT/docker/unsloth-paper"

# Single-turn evaluation
echo "Evaluating single-turn performance..."
docker-compose exec -T unsloth-paper-phi bash -c "cd paper_approach/scripts && \
    python eval_function_call.py \
        --model-id microsoft/phi-4-mini \
        --adapter-path ../adapters/phi_hermes_lora \
        --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
        --output-path ../eval_results/phi_single_turn.json" &

docker-compose exec -T unsloth-paper-llama bash -c "cd paper_approach/scripts && \
    python eval_function_call.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path ../adapters/llama_hermes_lora \
        --dataset-path ../../comparison_framework/test_splits/single_turn_test.jsonl \
        --output-path ../eval_results/llama_single_turn.json" &

wait

# Domain diversity evaluation
echo "Evaluating domain diversity..."
docker-compose exec -T unsloth-paper-phi bash -c "cd paper_approach/scripts && \
    python eval_function_call.py \
        --model-id microsoft/phi-4-mini \
        --adapter-path ../adapters/phi_hermes_lora \
        --dataset-path ../../comparison_framework/test_splits/domain_diverse_test.jsonl \
        --output-path ../eval_results/phi_domain_diverse.json" &

docker-compose exec -T unsloth-paper-llama bash -c "cd paper_approach/scripts && \
    python eval_function_call.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path ../adapters/llama_hermes_lora \
        --dataset-path ../../comparison_framework/test_splits/domain_diverse_test.jsonl \
        --output-path ../eval_results/llama_domain_diverse.json" &

wait

echo -e "${GREEN}✓ Evaluation complete${NC}"
echo ""

cd "$PROJECT_ROOT"

# Step 6: Generate comparison report
echo -e "${YELLOW}[7/7] Generating comparison report...${NC}"

python3 "$PROJECT_ROOT/comparison_framework/compare_approaches.py" \
    --fledgling-results "$PROJECT_ROOT/slm_swap/05_eval" \
    --paper-results "$PROJECT_ROOT/paper_approach/eval_results" \
    --output "$PROJECT_ROOT/comparison_framework/reports/phase1_comparison.json"

echo -e "${GREEN}✓ Comparison report generated${NC}"
echo ""

# Display results summary
echo "========================================================================"
echo -e "${GREEN}Phase 1 Experiment Complete!${NC}"
echo "========================================================================"
echo ""
echo "Results available at:"
echo "  - Comparison report: comparison_framework/reports/phase1_comparison.json"
echo "  - CSV exports: comparison_framework/reports/csv/"
echo "  - Phi-4-mini results: paper_approach/eval_results/phi_*.json"
echo "  - Llama results: paper_approach/eval_results/llama_*.json"
echo ""
echo "Next steps:"
echo "  1. Review comparison report: cat comparison_framework/reports/phase1_comparison.json | jq"
echo "  2. Check if success criteria met (see RUN_EXPERIMENTS.md)"
echo "  3. If successful, proceed to Phase 2 (scale up)"
echo ""
echo "To view results:"
echo "  python -m json.tool comparison_framework/reports/phase1_comparison.json"
echo ""
