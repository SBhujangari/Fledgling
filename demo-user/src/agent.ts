import './tracerSetup';
import { Agent } from '@mastra/core/agent';
import { z } from 'zod';
import { mapMastraTools, withMastraTracing } from '@fledgling/tracer';

import { registerAgentWithBackend } from './registerAgent';

const SAMPLE_LOCATIONS = [
  'San Francisco',
  'New York',
  'Berlin',
  'Singapore',
  'São Paulo',
  'Bengaluru',
  'Sydney',
] as const;

function getRandomLocation() {
  const index = Math.floor(Math.random() * SAMPLE_LOCATIONS.length);
  return SAMPLE_LOCATIONS[index];
}

//openai/gpt-5-nano

const QA_AGENT_ID = process.env.LANGFUSE_AGENT_ID || 'demo-agent';

const qaTools = {
  getLocation: {
    description: 'Returns the user\'s current location',
    parameters: z.object({
      reason: z
        .string()
        .describe('Summary of why you need a location (e.g., to contextualize recommendations).'),
    }),
    execute: async () => {
      const location = getRandomLocation();
      return { location };
    },
  },
} as const;

const QA_INSTRUCTIONS = `You are a reasoning assistant. Reasoning Effort: High. Think hard and a lot when given a problem and try to employ a COT approach.
1. Think through problem explicitly and break it down into smaller subtasks.
2. Call the tool 'getLocation' to get the user's location if relevant to their request.
3. After using tools, synthesize a concise final answer in 2-3 sentences referencing the retrieved location when relevant.
Return the final answer as normal assistant text (no tool syntax).`;

const rawQaAgent = new Agent({
  name: 'qa-agent',
  instructions: QA_INSTRUCTIONS,
  model: 'openai/gpt-5-nano',
  tools: qaTools,
});

setTimeout(() => {
  void registerAgentWithBackend({
    id: QA_AGENT_ID,
    name: 'QA Agent',
    taskDescription: 'Answers user questions with grounded recommendations',
    instructions: QA_INSTRUCTIONS,
    originalLLM: 'openai/gpt-5-nano',
    tags: ['demo', 'qa'],
    tools: mapMastraTools(QA_AGENT_ID, qaTools),
  });
}, 5000);

// Wrap the agent so all runs, LLM generations, and tool calls automatically log to Langfuse.
export const qaAgent = withMastraTracing(rawQaAgent, {
  agentId: QA_AGENT_ID,
  tags: ['demo', 'qa'],
});
