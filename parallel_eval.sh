#!/bin/bash
# Parallel evaluation across 4 GPUs for 4x speedup
# Each GPU processes different examples - no overlap

MODEL_ID="unsloth/llama-3.1-8b-instruct-bnb-4bit"
ADAPTER="slm_swap/04_ft/adapter_llama_hermes"
DATASET="paper_approach/datasets/hermes_test.jsonl"
OUTPUT_DIR="paper_approach/eval_results"

mkdir -p "$OUTPUT_DIR"

# Count total examples
TOTAL=$(wc -l < "$DATASET")
echo "Total examples: $TOTAL"

# Split into 4 chunks (one per GPU)
CHUNK_SIZE=$((TOTAL / 4))
echo "Chunk size per GPU: $CHUNK_SIZE"

# Split dataset
split -l "$CHUNK_SIZE" -d "$DATASET" /tmp/eval_chunk_

# Launch 4 parallel processes, one per GPU
for GPU in 0 1 2 3; do
    CHUNK_FILE="/tmp/eval_chunk_0${GPU}"
    OUTPUT_FILE="${OUTPUT_DIR}/llama_hermes_eval_gpu${GPU}.json"

    echo "Starting GPU $GPU with chunk $CHUNK_FILE"

    CUDA_VISIBLE_DEVICES=$GPU python docker/unsloth-paper/scripts/eval_function_call.py \
        --model-id "$MODEL_ID" \
        --adapter-path "$ADAPTER" \
        --dataset-path "$CHUNK_FILE" \
        --output-path "$OUTPUT_FILE" \
        --batch-size 1 \
        > "${OUTPUT_DIR}/gpu${GPU}.log" 2>&1 &
done

echo "Launched 4 parallel evaluation processes"
echo "Waiting for completion..."

# Wait for all background jobs
wait

echo "All evaluations complete!"
echo "Merging results..."

# Merge results
python3 - <<'EOF'
import json
import glob

results = []
for i in range(4):
    path = f"paper_approach/eval_results/llama_hermes_eval_gpu{i}.json"
    try:
        with open(path) as f:
            data = json.load(f)
            results.append(data)
    except:
        print(f"Warning: Could not load {path}")

# Merge metrics (average across GPUs)
if results:
    merged = {
        "valid_json_rate": sum(r.get("valid_json_rate", 0) for r in results) / len(results),
        "name_match_rate": sum(r.get("name_match_rate", 0) for r in results) / len(results),
        "args_exact_match_rate": sum(r.get("args_exact_match_rate", 0) for r in results) / len(results),
        "args_field_f1": sum(r.get("args_field_f1", 0) for r in results) / len(results),
        "total_examples": sum(r.get("total_examples", 0) for r in results),
    }

    with open("paper_approach/eval_results/llama_hermes_eval_merged.json", "w") as f:
        json.dump(merged, f, indent=2)

    print("\n" + "="*60)
    print("FINAL RESULTS (Merged)")
    print("="*60)
    print(f"Total examples: {merged['total_examples']}")
    print(f"Valid JSON rate: {merged['valid_json_rate']:.1%}")
    print(f"Name match rate: {merged['name_match_rate']:.1%}")
    print(f"Args exact match: {merged['args_exact_match_rate']:.1%}")
    print(f"Args F1 score: {merged['args_field_f1']:.1%}")
    print("="*60)
EOF

echo "Results saved to: paper_approach/eval_results/llama_hermes_eval_merged.json"
