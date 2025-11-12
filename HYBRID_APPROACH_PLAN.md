# Hybrid Approach: Fine-tuning + Auto Context Engineering

## Executive Summary

This plan combines:
1. **Fine-tuned Llama-3.1-8B** (orchestrator) + **Fine-tuned Phi-4** (sub-agents)
2. **Automatic Context Engineering** for continuous prompt optimization
3. **Multi-call Agentic Workflow** integration with automatic training data capture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-CALL WORKFLOW                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Orchestrator Agent (Llama-3.1-8B Fine-tuned)           │  │
│  │  - Handles complex multi-turn conversations             │  │
│  │  - Routes to appropriate sub-agents                     │  │
│  │  - Aggregates results from multiple calls               │  │
│  └────┬─────────────────────────────────────────────────┬───┘  │
│       │                                                  │      │
│       v                                                  v      │
│  ┌─────────────────┐                          ┌──────────────┐ │
│  │  Sub-Agent 1    │                          │ Sub-Agent N  │ │
│  │  (Phi-4 FT)     │      ...                 │ (Phi-4 FT)   │ │
│  │  - Fast         │                          │ - Efficient  │ │
│  │  - Specialized  │                          │ - Focused    │ │
│  └─────────────────┘                          └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             v
        ┌────────────────────────────────────────┐
        │   Auto Context Engineering Layer       │
        │   - Optimizes prompts per agent        │
        │   - Learns from failures               │
        │   - A/B tests template variants        │
        └────────────────────────────────────────┘
                             │
                             v
        ┌────────────────────────────────────────┐
        │   Automatic Training Data Capture      │
        │   - Records LLM decisions              │
        │   - Generates fine-tuning datasets     │
        │   - Continuous model improvement       │
        └────────────────────────────────────────┘
```

## Component Breakdown

### 1. Llama-3.1-8B (Orchestrator)

**Why Llama over Phi for Orchestrator?**
- Superior multi-turn conversation handling
- Better context retention (8K+ tokens)
- Native function calling understanding
- More robust for complex decision-making

**Training Configuration:**
```python
Model: meta-llama/Llama-3.1-8B-Instruct
LoRA: r=32, alpha=64, dropout=0.05
Dataset: Hermes Function Calling v1 (100 → 1000 examples)
Context Engineering: Enabled with error-aware templates
```

**Responsibilities:**
1. Parse user requests across multiple turns
2. Decide which sub-agent(s) to call
3. Format function calls with correct parameters
4. Aggregate responses from multiple sub-agents
5. Handle error recovery and retries

### 2. Phi-4-mini (Sub-Agents)

**Why Phi for Sub-Agents?**
- 2x faster inference than Llama
- Lower memory footprint (can run multiple in parallel)
- Cost-effective for high-volume simple tasks
- Sufficient for specialized, focused tasks

**Training Configuration:**
```python
Model: microsoft/phi-4-mini
LoRA: r=16, alpha=32, dropout=0.05
Dataset: Domain-specific subsets of Hermes
Context Engineering: Task-specific templates
```

**Specializations:**
- **Data Retrieval Agent**: Database queries, API calls
- **Processing Agent**: Data transformation, filtering
- **Validation Agent**: Schema validation, error checking
- **Formatting Agent**: Output formatting, serialization

### 3. Auto Context Engineering

**Framework:** `slm_swap/auto_context_engineering.py`

**How It Works:**

1. **Error Pattern Detection**
   ```python
   # Automatically identifies:
   - Markdown wrapper errors (```json ... ```)
   - Parameter name mismatches (query vs q)
   - Format violations (self-closing XML tags)
   - Missing required fields
   ```

2. **Template Evolution**
   ```python
   # Generates improved prompts like:
   "CRITICAL: Output ONLY raw JSON. NO markdown."
   "Start with <tool_call name='...'>. NO self-closing tags."
   "Use parameter 'q' not 'query'. Check signature."
   ```

3. **A/B Testing & Selection**
   ```python
   # Tracks performance per template:
   - Success rate: 87% → 94% (improved)
   - Usage count: 150 trials
   - Auto-selects best performing template
   ```

4. **Continuous Learning**
   ```python
   # Updates from workflow results:
   for result in workflow_results:
       if result.failed:
           analyze_error_patterns()
           generate_corrective_template()
           add_to_template_pool()
   ```

**Integration with Fine-tuning:**
- Fine-tuning provides **general capability**
- Context engineering provides **task-specific optimization**
- Together: Better than either alone

### 4. Multi-Call Workflow Integration

**Automatic Data Capture from LLM Workflows:**

```python
# Example: User runs existing LLM workflow
class WorkflowTracer:
    """Captures LLM decisions for SLM training"""

    def trace_call(self, agent_name, input_data, llm_response):
        """
        Captures every LLM call in workflow:
        - Agent type (orchestrator vs sub-agent)
        - Input query/context
        - Tool name and parameters
        - LLM's function call decision
        - Execution result (success/failure)
        """
        training_example = {
            "conversations": [
                {"from": "system", "value": self.get_agent_system_prompt(agent_name)},
                {"from": "human", "value": input_data},
                {"from": "gpt", "value": llm_response}
            ],
            "metadata": {
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "success": self.validate_execution(llm_response)
            }
        }

        self.append_to_dataset(training_example)

    def generate_fine_tune_dataset(self, min_examples=100):
        """
        Once enough examples collected:
        1. Filter for successful calls
        2. Add hard negatives (common failures)
        3. Balance across agent types
        4. Format for Hermes/Llama training
        """
        return self.create_balanced_dataset()
```

**Workflow Example:**

```python
# User's existing LLM agentic workflow
orchestrator_llm = AzureOpenAI(...)  # GPT-4

# 1. User query comes in
query = "Analyze sales data for Q4 and generate report"

# 2. Orchestrator decides sub-tasks (CAPTURED)
orchestrator_decision = orchestrator_llm.call(
    query=query,
    tools=available_tools
)
# → Captured: {"from": "gpt", "value": "<tool_call name='query_database'>..."}

# 3. Sub-agent executes (CAPTURED)
sub_agent_llm = AzureOpenAI(...)  # GPT-4
result = sub_agent_llm.call(
    task=orchestrator_decision.extracted_task
)
# → Captured: {"from": "gpt", "value": "SELECT * FROM sales..."}

# 4. After 100+ captures → Train SLM
train_hybrid_llama(captured_dataset)

# 5. Replace LLM with SLM
orchestrator_slm = load_fine_tuned_llama()
sub_agent_slm = load_fine_tuned_phi()
```

## Implementation Roadmap

### Phase 1: Foundation (Current)
**Status:** ✅ In Progress

- [x] Validate existing Fledgling SLM metrics
- [x] Identify gaps (toolcall 0% → needs fix)
- [x] Create auto context engineering framework
- [ ] Train Llama-3.1-8B on Hermes dataset
- [ ] Train Phi-4 on Hermes dataset

**Commands:**
```bash
# Train Llama (Orchestrator)
cd /home/gabriel/Desktop/AI_ATL25
CUDA_VISIBLE_DEVICES=0,1 python3 slm_swap/train_hybrid_llama.py

# Train Phi (Sub-agent)
cd /home/gabriel/Desktop/AI_ATL25
CUDA_VISIBLE_DEVICES=2,3 python3 docker/unsloth-paper/scripts/train_simple.py
```

### Phase 2: Multi-Call Integration (Next)
**Goal:** Capture existing workflow data automatically

**Tasks:**
1. **Workflow Tracer Implementation**
   ```python
   # File: slm_swap/workflow_tracer.py
   - Langfuse integration for call tracing
   - Automatic dataset generation from traces
   - Success/failure labeling
   ```

2. **Orchestrator-SubAgent Protocol**
   ```python
   # Define communication format:
   {
       "orchestrator_decision": {
           "sub_agents": ["retrieval", "processing"],
           "parallel": true,
           "aggregation_strategy": "merge"
       },
       "sub_agent_calls": [
           {"agent": "retrieval", "input": "...", "output": "..."},
           {"agent": "processing", "input": "...", "output": "..."}
       ]
   }
   ```

3. **Auto Fine-tuning Pipeline**
   ```python
   # Continuous learning loop:
   while workflow_running:
       capture_llm_calls()
       if dataset_size >= MIN_EXAMPLES:
           retrain_slm()
           evaluate_slm_vs_llm()
           if slm_performance >= THRESHOLD:
               swap_to_slm()
   ```

### Phase 3: Production Deployment
**Goal:** Seamless LLM → SLM transition

**Tasks:**
1. **Gradual Rollout**
   - 10% traffic → SLM (monitor)
   - If success rate ≥ 90% → 50% traffic
   - If success rate ≥ 95% → 100% SLM

2. **Fallback Mechanism**
   ```python
   try:
       result = orchestrator_slm.call(query)
       if confidence < THRESHOLD:
           result = orchestrator_llm.call(query)  # Fallback
   except:
       result = orchestrator_llm.call(query)
   ```

3. **Cost Monitoring**
   ```python
   # Track savings:
   llm_cost_per_1k_tokens = $0.03
   slm_cost_per_1k_tokens = $0.0001  # Local inference
   # → 300x cost reduction
   ```

## Expected Performance

### Baseline (Current Fledgling)
```
Structured JSON: 88% valid, 4% exact match, 32% F1
Toolcall XML: 0% valid (complete failure)
Gap to LLM: -28% F1 (structured), -74% F1 (toolcall)
```

### Target (Hybrid Approach)
```
Llama-3.1-8B + Context Engineering:
  Structured JSON: 98% valid, 85% exact match, 90% F1
  Toolcall XML: 95% valid, 90% exact match, 88% F1
  Gap to LLM: -5% F1 (acceptable for cost savings)

Phi-4 + Context Engineering (sub-agents):
  Specialized tasks: 92% F1
  General tasks: 85% F1
```

### Multi-Call Workflow Performance
```
Orchestrator (Llama-3.1-8B):
  - Multi-turn accuracy: 88%
  - Sub-agent routing: 94%
  - Result aggregation: 91%

End-to-End Workflow:
  - Single call: 90% success
  - Multi-call (2-3 steps): 85% success
  - Complex workflow (4+ steps): 78% success
```

## Key Advantages

### 1. Fine-tuning (Primary Driver)
- **Task-specific knowledge**: Learns exact parameter names, formats
- **Pattern recognition**: Generalizes across similar queries
- **Efficiency**: Smaller models can match LLM on narrow tasks

### 2. Auto Context Engineering (Multiplier)
- **Error recovery**: Catches and fixes format mistakes
- **Continuous improvement**: Gets better with usage
- **Zero-shot boost**: Helps even without retraining

### 3. Hybrid Model Strategy
- **Llama for complexity**: Orchestrator handles hard decisions
- **Phi for speed**: Sub-agents execute simple tasks fast
- **Cost optimization**: Right model for right task

### 4. Multi-Call Workflow Support
- **Automatic data capture**: No manual dataset creation
- **Real-world training**: Learns from actual workflow usage
- **Seamless transition**: LLM → SLM without workflow changes

## Next Steps

1. **Immediate** (Tonight):
   ```bash
   # Start Llama training
   ./slm_swap/train_hybrid_llama.py
   ```

2. **Tomorrow**:
   - Evaluate Llama vs Phi vs Fledgling
   - Implement workflow tracer prototype
   - Test multi-call orchestration

3. **This Week**:
   - Integrate with existing agentic workflow
   - Capture 100+ LLM workflow examples
   - Retrain with real workflow data
   - Measure LLM → SLM performance gap

4. **This Month**:
   - Deploy SLM in shadow mode (monitoring)
   - Gradual rollout (10% → 50% → 100%)
   - Measure cost savings and performance

## Success Criteria

- [ ] Llama-3.1-8B achieves ≥85% F1 on function calling
- [ ] Phi-4 achieves ≥80% F1 on specialized sub-tasks
- [ ] Multi-call workflow success rate ≥80%
- [ ] Auto context engineering improves performance by ≥10%
- [ ] End-to-end cost reduction ≥200x vs LLM
- [ ] Seamless integration with existing workflows

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SLM fails on edge cases | Fallback to LLM for low-confidence predictions |
| Training data quality | Validate captured examples before fine-tuning |
| Context engineering overfits | A/B test templates, maintain diverse pool |
| Multi-call coordination errors | Comprehensive testing suite for workflows |
| Model degradation over time | Continuous monitoring, periodic retraining |

## Conclusion

This hybrid approach combines the **best of all worlds**:
- Fine-tuning provides task-specific capability
- Context engineering provides continuous optimization
- Llama provides orchestration intelligence
- Phi provides efficient sub-agent execution
- Multi-call integration enables real-world deployment

**Result**: Production-ready SLM system that can replace LLM workflows at 200-300x lower cost while maintaining ≥80% performance.
