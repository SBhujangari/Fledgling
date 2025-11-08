AGENT.md — SLM Swap v0 (Evaluation-First, Minimal)
Goal

Decide if a local SLM can replace a hosted LLM for:

Track A — Structured JSON output

Track B — Tool-calling (single tool with JSON args)

Only fine-tune if the SLM underperforms the hosted LLM.

Constraints

OS: Ubuntu

Hosted LLM (teacher for baseline eval): Azure AI (OpenAI-compatible chat deployment of an open-source instruct model)

Local SLM (student): downloaded checkpoint (default path: slm_swap/models/qwen2.5-7b-instruct)

Fine-tuning: Unsloth QLoRA (only if needed)

Langfuse: MANDATORY — one trace per eval run and per train run (at minimum start/end + paths/metrics)

No fallbacks, routers, SAG/FTP, schema validators, or judges in v0

Repo Layout
slm_swap/
  02_dataset/
    structured/{train,val,test}.jsonl
    toolcall/{train,val,test}.jsonl
  04_ft/
    adapter_structured/
    adapter_toolcall/
  05_eval/
  models/                      # local SLM checkpoint
  agent.md
  README.md
Environment

AZURE_ENDPOINT (https://<endpoint>/openai/deployments/<deployment>)

AZURE_API_KEY

AZURE_API_VERSION (e.g., 2024-02-15-preview)

SLM_MODEL_PATH (default slm_swap/models/qwen2.5-7b-instruct)

Langfuse (required): LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

Data (row format)

Two tracks; each row has exactly prompt and completion.

Structured
prompt: concise instruction + user text.
completion: pure JSON (no prose).

Tool-calling
prompt: concise instruction + tool signature + user text.
completion: exactly one wrapper
<tool_call name="TOOL_NAME">{ ...json args... }</tool_call>.

Initial source: small splits derived from Salesforce/xlam-function-calling-60k. Keep splits small and consistent.

Models

Hosted LLM (Azure): open-source instruct model via Azure chat completions, temperature=0.

SLM (local): load from SLM_MODEL_PATH in 4-bit, greedy decoding.

Evaluation (baseline first)

Run on the test split for each track and each model. Record a Langfuse trace per run.

Structured metrics

json_valid_rate: output parses as JSON.

exact_match_rate: canonical JSON equality vs gold (sorted keys, compact separators).

Tool-calling metrics

valid_call_rate: wrapper present and args parse as JSON.

name_match_rate: tool name equals gold.

args_exact_rate: canonicalized args equal gold.

Metrics are written as small JSON files under 05_eval/.

Decision Rule (gate to fine-tune)

Compare Azure vs SLM per track.
If SLM is worse than Azure on any core metric by margin delta > 0, set FINE_TUNE=1 for that track; otherwise skip fine-tuning. Record the decision in a Langfuse event.

Fine-Tuning (only if needed)

Method: Unsloth QLoRA, base = SLM_MODEL_PATH.

Fixed settings: 4-bit load, LoRA r=64, alpha=16, dropout=0.1, lr=2e-4, 1 epoch, seq_len≈2048.

Input: train.jsonl / val.jsonl with the same prompt → completion format.

Outputs: LoRA adapters:

04_ft/adapter_structured

04_ft/adapter_toolcall

Re-run the same evaluation on test after training (new Langfuse trace).

Integration (Mastra)

Add a thin Mastra client that builds the same messages used in evaluation, calls the local SLM (and loads the adapter if present), and returns raw text. No routing or fallback in v0.

Roadmap (Phase II — Full Auto LLM→SLM)

In scope for the full project, implemented after v0 smoke test:

Teacher trajectories: instrument the hosted LLM agent to log one CodeAct-style trajectory per request (Thought, Action/Code, Observation, final answer).

Observation masking: prepare student training texts where Observations are excluded from loss; train only on reasoning + action tokens.

Automatic filtering: keep only trajectories that pass automatic correctness checks for their dataset/task.

Optional FTP/SAG: reserve for later; not in v0.

Auto dataset builder: convert captured trajectories into train/val/test JSONLs for both tracks.

End-to-end traceability: Langfuse spans for capture, filter, build, train, eval.

(Optional later) Router: SLM default with strict validators and teacher fallback.
Deliverables for Phase II: capture shim, builder, masked-loss FT, automated eval, and docs.

Definition of Done (v0)

Local SLM downloaded to models/ and loads.

Tiny structured/toolcall datasets exist under 02_dataset/ with test splits populated.

Baseline evaluations completed for Azure and SLM; metrics saved in 05_eval/; Langfuse traces present.

Decision produced per track (fine-tune or skip) and logged to Langfuse.

If fine-tuned: adapters saved under 04_ft/ and post-FT evaluation re-run with traces.

agent.md and README.md updated and committed.
