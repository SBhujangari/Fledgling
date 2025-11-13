# Integration TODO

1. **Langfuse-First Capture**
   - [ ] Enforce `withMastraTracing` + `LANGFUSE_*` across every agent (demo + production).
   - [ ] Extend `/api/traces` payload with schema/tool validation status and Langfuse deep links.

2. **UI Observability Hooks**
   - [ ] Wire the original frontend to `/api/traces`, `/api/agents`, `/api/tools`, `/api/training/status`.
   - [ ] Add dashboards for Langfuse trace counts, recent runs, and parity metrics.

3. **Pipeline Triggers**
   - [ ] Create backend jobs for dataset refresh (`langfuse_dataset.py`), evals (`eval.py`, `compare.py`), and fine-tuning (`train_unsloth.py`).
   - [ ] Stream stdout/stderr + Langfuse trace IDs back to the frontend.

4. **Decision & Swap Logic**
   - [ ] Persist compare decisions, expose ParityBench reports, and gate the “Promote SLM” action.

5. **Runtime Swap**
   - [ ] Update agents to load the promoted SLM adapter with deterministic seeds and teacher fallback.

6. **Docs / SOP**
   - [ ] Merge `AGENT.md`, `README.md`, and pipeline docs into a single “Parity → Swap Playbook”.

7. **Automated NHITL Loop**
   - [ ] Schedule recurring capture → dataset → eval → fine-tune cycles, logging every step in Langfuse.

8. **Frontend Stabilization**
   - [ ] Diagnose `/metrics` returning a blank page (confirm `/api/metrics/comparison` fetch, add loading/error states, and ensure the new comparison types match the backend payload).
   - [ ] Rework the “Import client workflow” form so a PM can upload a LangGraph or n8n workflow file/folder, store it server-side, and immediately register the agent + workflow metadata.
   - [ ] Embed the Langfuse agent graph locally by rendering cached trace JSON (no external Langfuse API), keeping the existing visual style but sourcing data from the backend’s stored samples.
