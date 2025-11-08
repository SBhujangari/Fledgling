# SLM Swap v0 (Evaluation-First, Minimal)

## Goal
Decide whether a local SLM can replace the hosted Azure LLM for two tracks:
- **Track A — Structured JSON output**
- **Track B — Tool-calling (single tool with JSON args)**

Only fine-tune the SLM if it underperforms the hosted LLM.

## Constraints
- OS: Ubuntu
- Hosted LLM (teacher for baseline eval): Azure AI (OpenAI-compatible chat deployment of an open-source instruct model)
- Local SLM (student): downloaded checkpoint (default path: `slm_swap/models/phi-4-mini`)
- Fine-tuning: Unsloth QLoRA (only if needed)
- Langfuse: **MANDATORY** — at minimum one trace per eval run and per train run (record start/end + dataset/metrics paths)
- No fallbacks, routers, SAG/FTP, schema validators, or judges in v0

## Repo Layout
```
slm_swap/
  02_dataset/
    structured/{train,val,test}.jsonl
    toolcall/{train,val,test}.jsonl
  04_ft/
    adapter_structured/
    adapter_toolcall/
  05_eval/
  models/
  agent.md
  README.md
```

## Environment
- `AZURE_ENDPOINT` — `https://<endpoint>/openai/deployments/<deployment>`
- `AZURE_API_KEY`
- `AZURE_API_VERSION` (e.g., `2024-02-15-preview`)
- `SLM_MODEL_PATH` (default `slm_swap/models/qwen2.5-7b-instruct`)
- Langfuse (required): `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Data (row format)
Two tracks; each row has exactly `prompt` and `completion`.

**Structured**
- `prompt`: concise instruction + user text describing the API call.
- `completion`: pure JSON (no prose) describing the tool name + arguments.

**Tool-calling**
- `prompt`: concise instruction + tool signature + user text.
- `completion`: exactly one wrapper `<tool_call name="TOOL_NAME">{ ...json args... }</tool_call>`.

Initial source: tiny splits derived from `Salesforce/xlam-function-calling-60k` via `prepare_data.py` (uses the gated `xlam_function_calling_60k.json`). Keep splits small and consistent; drop any Observation/tool-output text for tool-calling completions.

## Models
- Hosted LLM (Azure): open-source instruct model via Azure chat completions, temperature = 0.
- SLM (local): load from `SLM_MODEL_PATH` (default Phi-4-mini) in 4-bit, greedy decoding.

## Evaluation (baseline first)
Run on the test split for each track and each model. Record a Langfuse trace per run.

**Structured metrics**
- `json_valid_rate`: output parses as JSON.
- `exact_match_rate`: canonical JSON equality vs gold (sorted keys, compact separators).

**Tool-calling metrics**
- `valid_call_rate`: wrapper present and args parse as JSON.
- `name_match_rate`: tool name equals gold.
- `args_exact_rate`: canonicalized args equal gold.

Metrics live under `05_eval/` as small JSON files.

## Decision Rule (gate to fine-tune)
Compare Azure vs SLM per track. If the SLM is worse than Azure on any core metric by a margin `delta > 0`, set `FINE_TUNE=1` for that track; otherwise skip fine-tuning. Record the decision in a Langfuse event.

## Fine-Tuning (only if needed)
- Method: Unsloth QLoRA, base = `SLM_MODEL_PATH`.
- Fixed settings: 4-bit load, LoRA `r=64`, `alpha=16`, `dropout=0.1`, `lr=2e-4`, `epochs=1`, `seq_len≈2048`, launched with multi-GPU parallelism (4×3090 recommended via `accelerate launch --config_file accelerate_config/phi4_4gpu.yaml`).
- Input: `train.jsonl` / `val.jsonl` with the same prompt → completion format.
- Outputs: LoRA adapters (`04_ft/adapter_structured` and `04_ft/adapter_toolcall`).
- Re-run the same evaluation on test after training (new Langfuse trace).

## Integration (Mastra)
Add a thin Mastra client that builds the same messages used in evaluation, calls the local SLM (and loads the adapter if present), and returns raw text. No routing or fallback in v0.

## Roadmap — Phase II (teacher trajectories + observation masking + auto LLM→SLM)
Planned after the v0 smoke test:
- **Teacher trajectories**: instrument the hosted LLM agent to log CodeAct-style trajectories per request (Thought, Action/Code, Observation, final answer).
- **Observation masking**: prepare student training texts where Observation tokens are excluded from loss; train only on reasoning + action tokens.
- **Automatic LLM→SLM pipeline**: capture, filter, and build datasets from trajectories with correctness filters; convert them into structured/tool-call JSONLs.
- **Traceability**: Langfuse spans for capture, filtering, building, training, and evaluation.
- **Optional later**: FTP/SAG modules and router with validator + teacher fallback.
- **Deliverables**: capture shim, builder, masked-loss FT, automated eval, and docs.

*Phase II is documented but not implemented in this pass.*

## Definition of Done (v0)
1. Local SLM downloaded to `models/` and loads.
2. Tiny structured/toolcall datasets exist under `02_dataset/` with populated test splits.
3. Baseline evaluations completed for Azure and SLM; metrics saved in `05_eval/`; Langfuse traces exist.
4. Decision produced per track (fine-tune or skip) and logged to Langfuse.
5. If fine-tuned: adapters saved under `04_ft/` and post-FT evaluation re-run with traces.
6. `agent.md` and `README.md` updated and committed.
