AGENT.md — LLM to SLM Swap in Agent Architecture

Note: Claude will NOT include itself in git commits


Goal

Prove—using real agent traces—that our specialist SLM can replace the hosted LLM on:

Track A — Structured JSON output  
Track B — Tool-calling (single tool with JSON args)

We only touch fine-tuning if the SLM underperforms, and every promotion requires documented parity deltas (tool accuracy, schema conformance, latency/cost). The win condition is **SLM ≥ LLM** on acceptance metrics plus a deterministic swap kit.

Agent-Parity Stance

- Capture live agent traces first (via `withMastraTracing()` + Langfuse), not synthetic prompts.
- Convert those traces into tiny structured/tool-call splits with `python langfuse_dataset.py --agent-id <...> --output-root python-pipeline/slm_swap/02_dataset`.
- Run eval → compare → train loops with Langfuse traces attached so we can replay every decision.
- Stay vendor-agnostic on training backends; this repo uses Unsloth QLoRA but anything pluggable works.

Constraints

OS: Ubuntu

commit early and often

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
  langfuse_dataset.py          # convert Langfuse traces into dataset splits
  agent.md
  README.md
Environment

AZURE_ENDPOINT (https://<endpoint>/openai/deployments/<deployment>)

AZURE_API_KEY

AZURE_API_VERSION (e.g., 2024-02-15-preview)

SLM_MODEL_PATH (default slm_swap/models/qwen2.5-7b-instruct)

> When running inside `slm_swap/`, point `SLM_MODEL_PATH` at `models/<checkpoint>` (or use an absolute path). Adding another `slm_swap/` prefix creates a nonexistent `slm_swap/slm_swap/...` folder and the loader will fail fast.

Langfuse (required): LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

Determinism and Reproducibility

All evaluation and training runs enforce strict determinism:

Random seeds: Fixed at 42 for Python random, NumPy, PyTorch (CPU/CUDA), and transformers.

SLM inference: Model in eval mode (dropout disabled), greedy decoding (do_sample=False), temperature=0, top_p=1.0, top_k=0, CUDA deterministic operations enabled.

Azure LLM: temperature=0, seed=42 passed to API for reproducible sampling.

Environment: PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms=True.

This ensures bit-for-bit identical outputs across runs given the same inputs, hardware, and software versions.

Data (row format)

Two tracks; each row has exactly prompt and completion.

Structured
prompt: concise instruction + user text.
completion: pure JSON (no prose).

Tool-calling
prompt: concise instruction + tool signature + user text.
completion: exactly one wrapper
<tool_call name="TOOL_NAME">{ ...json args... }</tool_call>.

`langfuse_dataset.py` emits both tracks automatically from Langfuse traces (prompts already include the necessary formatting + deterministic completions).

Need test data fast? Run `python python-pipeline/slm_swap/dummy_agent_workflow.py --out-json storage/dummy_langfuse_traces.jsonl --agent-id dummy-agent` followed by `python python-pipeline/slm_swap/langfuse_dataset.py --trace-json storage/dummy_langfuse_traces.jsonl --agent-id dummy-agent --output-root python-pipeline/slm_swap/02_dataset/dummy_langfuse`.

Initial source: small splits derived from Salesforce/xlam-function-calling-60k. Keep splits small and consistent.

For the first smoke tests, run `python run_first_tests.py` inside `slm_swap/` to resample 50 **distinct** prompts per track and automatically launch the live evaluation dashboard for Structured and Toolcall runs. This keeps the datasets tiny while still logging Langfuse traces.

Models

Hosted LLM (Azure): open-source instruct model via Azure chat completions, temperature=0, seed=42 for determinism.

SLM (local): load from SLM_MODEL_PATH in 8-bit, greedy decoding (do_sample=False, temperature=0, top_p=1.0, top_k=0), eval mode, seed=42 for determinism.

Evaluation (baseline first)

Run on the test split for each track and each model. Record a Langfuse trace per run.

SLM evaluation uses batched inference (default batch_size=8) to process multiple examples in parallel across 4 GPUs for maximum throughput. The model is tensor-parallelized across GPUs (layers distributed), and batching enables concurrent processing.

Use `live_compare.py` when you need real-time visibility into Azure vs SLM metrics; it streams per-example progress, logs Langfuse traces with `mode=live`, and can optionally persist the same JSON outputs as `eval.py`.

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

Fixed settings: 8-bit load, LoRA r=64, alpha=16, dropout=0.1, lr=2e-4, 1 epoch, seq_len≈2048.

Input: train.jsonl / val.jsonl with the same prompt → completion format.

Outputs: LoRA adapters:

04_ft/adapter_structured

04_ft/adapter_toolcall

Re-run the same evaluation on test after training (new Langfuse trace).

Multi-GPU Training Strategies

Two options for training both tracks:

Sequential (train_sequential.sh): Train structured then toolcall using all 4 GPUs per model. Lower communication overhead, better GPU utilization per model, but 2x wallclock time.

Parallel (train_parallel.sh): Train both models simultaneously using 2 GPUs each. Halves wallclock time but may have slightly slower per-model training.

For small datasets (<10k examples), parallel training typically completes faster overall despite lower per-GPU efficiency. Run ./train_parallel.sh for default parallel execution or ./train_sequential.sh for sequential.

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
