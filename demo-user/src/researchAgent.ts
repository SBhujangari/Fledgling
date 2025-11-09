import './tracerSetup';
import { randomUUID } from 'node:crypto';
import { Agent } from '@mastra/core/agent';
import { z } from 'zod';
import { mapMastraTools, withMastraTracing } from '@fledgling/tracer';

import { registerAgentWithBackend } from './registerAgent';

const RESEARCH_AGENT_ID = process.env.LANGFUSE_RESEARCH_AGENT_ID || 'research-agent';

const researchTools = {
  fetchDataset: {
    description: 'Retrieves a dataset summary for a requested topic',
    parameters: z.object({
      topic: z.string().describe('Dataset topic or domain to retrieve'),
    }),
    execute: async ({ topic }: { topic: string }) => {
      return {
        topic,
        source: 'internal-dataset-api',
        rows: Math.floor(Math.random() * 5_000) + 1_000,
      };
    },
  },
  writeMemo: {
    description: 'Stores a short memo capturing the synthesized insight',
    parameters: z.object({
      insight: z.string().describe('Key takeaway or memo text'),
    }),
    execute: async ({ insight }: { insight: string }) => {
      return { id: randomUUID(), insight };
    },
  },
} as const;

const RESEARCH_INSTRUCTIONS = `You are a research strategist. Evaluate the user's hypothesis, gather supporting info via tools, and return a structured recommendation with risks and next steps.`;

const rawResearchAgent = new Agent({
  name: 'research-agent',
  instructions: RESEARCH_INSTRUCTIONS,
  model: 'openai/gpt-5-medium',
  tools: researchTools,
});

setTimeout(() => {
  void registerAgentWithBackend({
    id: RESEARCH_AGENT_ID,
    name: 'Research Strategist',
    taskDescription: 'Brainstorms and evaluates product bets with dataset-backed reasoning',
    instructions: RESEARCH_INSTRUCTIONS,
    originalLLM: 'openai/gpt-4o',
    tags: ['demo', 'research'],
    tools: mapMastraTools(RESEARCH_AGENT_ID, researchTools),
  });
}, 5000);

export const researchAgent = withMastraTracing(rawResearchAgent, {
  agentId: RESEARCH_AGENT_ID,
  tags: ['demo', 'research'],
});
