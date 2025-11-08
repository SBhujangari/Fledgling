export interface ConversationMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface UsageStats {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
  cachedInputTokens?: number;
}

export interface ThoughtStep {
  type: 'thought';
  content: string;
  observationId?: string;
  timestamp?: string;
}

export interface ToolCallStep {
  type: 'tool_call';
  toolName: string;
  input: unknown;
  output?: unknown;
  status?: string;
  observationId?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface GenerationStep {
  type: 'generation';
  model?: string;
  prompt?: unknown;
  completion?: unknown;
  usage?: UsageStats;
  observationId?: string;
}

export type ParsedStep = ThoughtStep | ToolCallStep | GenerationStep;

export interface FinetuneSample {
  traceId: string;
  agentId: string;
  conversation: ConversationMessage[];
  steps: ParsedStep[];
  finalResponse?: string;
  usage?: UsageStats;
  metadata?: Record<string, unknown> | null;
}
