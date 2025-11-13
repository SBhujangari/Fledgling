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

  const payload: AgentExecutionPayload = {
    agentId: agent.id,
    agentName: agent.name,
    instructions: agent.instructions || agent.taskDescription,
    tools,
    prompt: request.prompt,
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
  const defaultSlmModel = 'meta-llama/llama-3.1-8b-instruct';

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
    const modelLabel = typeof config.model === 'string' ? config.model : null;
    return {
      label,
      model: modelLabel,
      provider: config.provider,
      result: { error: `No adapter registered for provider ${config.provider}` },
    };
  }

  const result = await adapter.run(config, payload);
  const modelLabel = typeof config.model === 'string' ? config.model : null;
  return {
    label: config.label ?? label,
    model: modelLabel,
    provider: config.provider,
    result,
    fallback: Boolean(config.metadata?.fallback),
  };
}
