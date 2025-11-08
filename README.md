# Fledgling
Making task inference cheaper and more reliable with a local Phi-4-mini SLM + Azure LLM baseline.

## Local SLM Download (Phi-4-mini)
```bash
huggingface-cli download microsoft/phi-4-mini \
  --local-dir slm_swap/models/phi-4-mini
```
The `slm_client` loads this checkpoint with 8-bit quantization, `device_map="auto"`, and `torch_dtype=torch.bfloat16`, so it automatically shards across four RTX 3090s when available. Evaluation uses batched inference (default batch_size=8) to process multiple examples in parallel across all 4 GPUs for maximum throughput.

## Environment Variables
```bash
export SLM_MODEL_PATH="slm_swap/models/phi-4-mini"
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
# optional local smoke tests
# export LANGFUSE_DISABLED=1

export AZURE_ENDPOINT="https://<resource>.openai.azure.com/openai/deployments/<deployment>"
export AZURE_DEPLOYMENT="<deployment>"
export AZURE_API_KEY="..."
export AZURE_API_VERSION="2024-05-01-preview"
```
The Azure endpoint must follow the conventional `/openai/deployments/<deployment>` form or an equivalent `models/chat/completions` endpoint that your resource exposes.

## Multi-GPU Fine-Tuning (Unsloth QLoRA)

### Quick Start
```bash
cd slm_swap

# Parallel training (2+2 GPUs) - recommended for small datasets, halves wallclock time
./train_parallel.sh

# Sequential training (4 GPUs per model) - maximum per-model speed
./train_sequential.sh
```

### Manual Launch
Launch via Accelerate (preferred) or Torchrun:
```bash
# Accelerate (4 processes, ZeRO-2 via accelerate_config/deepspeed_zero2.json)
accelerate launch --config_file accelerate_config/phi4_4gpu.yaml \
  train_unsloth.py --track structured \
  --train 02_dataset/structured/train.jsonl \
  --val 02_dataset/structured/val.jsonl \
  --out 04_ft/adapter_structured

# Torchrun equivalent
torchrun --nproc_per_node=4 train_unsloth.py --track toolcall \
  --train 02_dataset/toolcall/train.jsonl \
  --val 02_dataset/toolcall/val.jsonl \
  --out 04_ft/adapter_toolcall
```
`train_unsloth.py` wires gradient checkpointing, `ddp_find_unused_parameters=False`, and default DeepSpeed ZeRO-2 config (`slm_swap/accelerate_config/deepspeed_zero2.json`) so multi-GPU utilization is automatic.

## Repo Map
- `slm_swap/README.md` — full workflow (datasets, baseline evals, comparisons, fine-tune loop).
- `slm_swap/agent.md` — operator playbook detailing the evaluation-first plan.
- `slm_swap/accelerate_config/*.yaml|json` — launcher templates for 4×3090 training.
- `requirements.txt` — install once from repo root (`deepspeed`, `unsloth`, `trl`, etc.).

See `slm_swap/README.md` for end-to-end instructions covering dataset prep, Langfuse expectations, evaluations, and decision rules.

## Future Roadmap — Model Graders vs Teacher–Student
We may later layer in OpenAI’s model-grader-driven reinforcement fine-tuning (RFT) workflow that relies on explicit reward functions to shape reasoning behavior (inspiration: [OpenAI Cookbook example](https://cookbook.openai.com/examples/reinforcement_fine_tuning)). That approach differs from today’s teacher/student plan (`AGENT.md`) which prioritizes swapping in a local SLM, matching teacher metrics track-by-track, and only running Unsloth QLoRA when the SLM underperforms. For now this section is informational only—no grader-based automation is scheduled for implementation yet, but keeping the contrast documented helps if we decide to expand the roadmap.
