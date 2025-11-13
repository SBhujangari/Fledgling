import type { ToolRecord } from '../../types/persistence';
import type { UsageStats, FinetuneSample } from '../../types/finetune';

export interface ModelEndpointConfig {
  provider: 'openai' | 'mastra' | 'local' | (string & {});
  model: string | unknown;
  label?: string;
  temperature?: number;
  metadata?: Record<string, unknown>;
}

export interface ComparisonRequest {
  agentId: string;
  prompt: string;
  llmModel?: ModelEndpointConfig;
  slmModelOverride?: ModelEndpointConfig;
}

export interface AgentExecutionPayload {
  agentId: string;
  agentName: string;
  instructions: string;
  tools: ToolRecord[];
  prompt: string;
}

export interface AgentExecutionResult {
  output?: string;
  usage?: UsageStats;
  latencyMs?: number;
  traceId?: string;
  raw?: unknown;
  error?: string;
}

export interface ComparisonRunRecord {
  label: string;
  model: string | null;
  provider: string;
  result: AgentExecutionResult;
  fallback?: boolean;
}

export interface ComparisonResult {
  agentId: string;
  prompt: string;
  tools: ToolRecord[];
  runs: ComparisonRunRecord[];
  status: 'completed' | 'failed';
  traceSample?: FinetuneSample | null;
  originalModel?: string | null;
}
