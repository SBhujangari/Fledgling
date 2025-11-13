export interface ExampleWorkflow {
  id: string;
  title: string;
  summary: string;
  datasetPath: string;
  evaluationCommand: string;
  features: string[];
  codeSnippet: string;
  defaultAgent: {
    id: string;
    name: string;
    taskDescription: string;
    instructions: string;
    originalLLM: string;
    tags: string[];
  };
}

export const exampleWorkflows: ExampleWorkflow[] = [
  {
    id: 'structured-biz-entity',
    title: 'Structured Biz Entity LangGraph Agent',
    summary:
      'Demonstrates LangGraph planner/extractor pattern with Langfuse tracing and our structured JSON dataset. Mirrors the Langfuse cookbook agent while feeding examples into the auto fine-tune loop.',
    datasetPath: 'datasets/biz_entity_location_date_structure.jsonl',
    evaluationCommand:
      'python python-pipeline/slm_swap/run_dummy_pipeline.py --count 48 && python python-pipeline/slm_swap/eval.py --track structured --model-kind slm',
    features: [
      'LangGraph planner → extractor nodes instrumented via Langfuse traces',
      'Dataset seeding using Langfuse cookbook dataset guide',
      'Feeds structured JSON metrics into compare.py to decide fine-tune vs swap',
    ],
    codeSnippet: `import { START, END, StateGraph } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";

type BizEntityState = { messages: string[]; entity?: Record<string, string> };

const graph = new StateGraph<BizEntityState>({ channels: { messages: { value: (state, update) => [...state.messages, ...update], default: () => [] } } });

graph.addNode("planner", async (state) => {
  return { messages: [...state.messages, "Plan: extract entity, location, date"] };
});

graph.addNode("extractor", async (state) => {
  // Call LLM/tooling here; simplified for the cookbook example
  const entity = { entity: "Acme Robotics", location: "Tokyo", date: "2025-04-01" };
  return { messages: [...state.messages, JSON.stringify(entity)], entity };
});

graph.addEdge(START, "planner");
graph.addEdge("planner", "extractor");
graph.addEdge("extractor", END);

export const bizEntityWorkflow = graph.compile();`,
    defaultAgent: {
      id: 'cookbook-biz-entity',
      name: 'Cookbook Biz Entity Agent',
      taskDescription: 'Parses company/location/date triples from announcements.',
      instructions:
        'You extract entity, location, and ISO date fields following the structured JSON schema. Log every intermediate step to Langfuse.',
      originalLLM: 'gpt-4o-mini',
      tags: ['cookbook', 'structured', 'langgraph'],
    },
  },
  {
    id: 'toolcall-ops-agent',
    title: 'Tool-Calling Ops Agent',
    summary:
      'Sample workflow that replays the tool-calling evaluation dataset via Langfuse datasets API and runs compare.py to gate auto fine-tuning.',
    datasetPath: 'python-pipeline/slm_swap/02_dataset/toolcall/test.jsonl',
    evaluationCommand:
      'python python-pipeline/slm_swap/eval.py --track toolcall --model-kind slm --split test && python python-pipeline/slm_swap/compare.py --track toolcall --azure 05_eval/toolcall_azure_test.json --slm 05_eval/toolcall_slm_test.json --delta 0.01',
    features: [
      'Langfuse dataset upsert for tool-call schema (per cookbook datasets guide)',
      'Deterministic tool-call evaluation feeding ParityBench/compare.py',
      'Swap readiness summary surfaced in the Trace Console page',
    ],
    codeSnippet: `import { START, END, StateGraph } from "@langchain/langgraph";

type ToolState = { messages: string[]; toolCall?: { name: string; args: Record<string, unknown> } };

const graph = new StateGraph<ToolState>({ channels: { messages: { value: (state, update) => [...state.messages, ...update], default: () => [] } } });

graph.addNode("router", async (state) => ({ messages: [...state.messages, "Route request to calendar.getAvailability"] }));

graph.addNode("tool", async (state) => ({
  messages: [...state.messages, 'Invoked calendar.getAvailability'],
  toolCall: { name: "calendar.getAvailability", args: { date: "2025-05-06" } },
}));

graph.addEdge(START, "router");
graph.addEdge("router", "tool");
graph.addEdge("tool", END);

export const toolCallWorkflow = graph.compile();`,
    defaultAgent: {
      id: 'cookbook-toolcall',
      name: 'Cookbook Tool-Call Agent',
      taskDescription: 'Executes deterministic tool calls (calendar, ops, logistics) and logs every span.',
      instructions:
        'You must emit exactly one <tool_call> wrapper per request, match schema keys, and log traces to Langfuse before returning.',
      originalLLM: 'gpt-4o-mini',
      tags: ['cookbook', 'toolcall', 'langgraph'],
    },
  },
];
