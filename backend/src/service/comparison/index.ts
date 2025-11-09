import { getAgentById } from '../store/agentStore';
import { listToolsByIds } from '../store/toolStore';
import { fetchLatestTraceSample } from '../traceFetcher';
import { MastraComparisonAdapter } from './mastraAdapter';
import type { AgentComparisonAdapter } from './adapters';
import type {
  AgentExecutionPayload,
  ComparisonRequest,
  ComparisonResult,
  ModelEndpointConfig,
  ComparisonRunRecord,
} from './types';
import { openrouterClient, openaiClient } from '../../clients';

const adapters: AgentComparisonAdapter[] = [new MastraComparisonAdapter()];

function resolveAdapter(config: ModelEndpointConfig): AgentComparisonAdapter | null {
  return adapters.find((adapter) => adapter.supports(config)) ?? null;
}

export async function compareAgentModels(request: ComparisonRequest): Promise<ComparisonResult> {
  const agent = await getAgentById(request.agentId);
  if (!agent) {
    throw new Error(`Agent with id "${request.agentId}" not found`);
  }

  const tools = agent.toolIds?.length ? await listToolsByIds(agent.toolIds) : [];

  // Enhance prompt with autonomous execution instructions for demo
  const enhancedPrompt = tools.length > 0
    ? `${request.prompt}\n\nIMPORTANT: You are an autonomous agent. Do not ask the user any questions. Instead:\n- Reason through the problem step by step\n- Use available tools proactively to gather information\n- Make multiple tool calls as needed to solve the task\n- Synthesize information from tool results\n- Provide a complete solution without asking for clarification\n- Act decisively and independently`
    : request.prompt;

  const payload: AgentExecutionPayload = {
    agentId: agent.id,
    agentName: agent.name,
    instructions: agent.instructions || agent.taskDescription,
    tools,
    prompt: enhancedPrompt,
  };

  // Helper to resolve model string to model instance
  const resolveModel = (modelStr: string) => {
    // OpenAI models start with openai/
    if (modelStr.startsWith('openai/')) {
      const modelId = modelStr.replace('openai/', '');
      return openaiClient(modelId);
    }
    // Everything else goes through OpenRouter
    return openrouterClient(modelStr);
  };

  const llmConfig = request.llmModel ?? {
    provider: 'mastra',
    model: resolveModel('openai/gpt-5'),
    label: 'GPT-5 (baseline)',
    temperature: 0.2,
  };

  const hasCustomSlm = Boolean(agent.slmModel);
  const isValidSlmModel = hasCustomSlm && typeof agent.slmModel === 'string' && agent.slmModel.includes('/');
  const defaultSlmModel = 'openai/gpt-5';

  const slmConfig: ModelEndpointConfig = request.slmModelOverride ?? {
    provider: 'mastra',
    model: resolveModel(isValidSlmModel ? (agent.slmModel as string) : defaultSlmModel),
    label: `${agent.name} SLM`,
    temperature: 0.2,
    metadata: isValidSlmModel ? undefined : { fallback: true },
  };

  const runs = await Promise.all([
    runModel('Baseline', llmConfig, payload),
    runModel(agent.name, slmConfig, payload),
  ]);

  const hasSuccessfulRun = runs.some((run) => !run.result.error);
  const status: ComparisonResult['status'] = hasSuccessfulRun ? 'completed' : 'failed';
  const traceSample = await fetchLatestTraceSample(agent.id);

  return {
    agentId: agent.id,
    prompt: request.prompt,
    tools,
    runs,
    status,
    traceSample,
    originalModel: agent.originalLLM ?? null,
  };
}

async function runModel(
  label: string,
  config: ModelEndpointConfig,
  payload: AgentExecutionPayload,
): Promise<ComparisonRunRecord> {
  const adapter = resolveAdapter(config);

  if (!adapter) {
    return {
      label,
      model: config.model,
      provider: config.provider,
      result: { error: `No adapter registered for provider ${config.provider}` },
    };
  }

  const result = await adapter.run(config, payload);
  return {
    label: config.label ?? label,
    model: config.model,
    provider: config.provider,
    result,
    fallback: Boolean(config.metadata?.fallback),
  };
}
