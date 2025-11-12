export interface LangfuseObservationUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  inputCost?: number;
  outputCost?: number;
  totalCost?: number;
}

export interface LangfuseObservation {
  id: string;
  traceId?: string;
  name?: string;
  type?: string;
  status?: string;
  metadata?: Record<string, unknown> | null;
  parentObservationId?: string | null;
  startTime?: string | null;
  endTime?: string | null;
  input?: unknown;
  output?: unknown;
  model?: string;
  usage?: LangfuseObservationUsage | null;
}

export interface LangfuseTrace {
  id: string;
  name?: string;
  status?: string;
  metadata?: Record<string, unknown> | null;
  status?: string;
  timestamp?: string;
  createdAt?: string;
  updatedAt?: string;
  startTime?: string | null;
  endTime?: string | null;
  input?: unknown;
  output?: unknown;
  tags?: string[];
  latency?: number;
  totalCost?: number;
  observations?: LangfuseObservation[];
}

export interface AgentRun {
  traceId: string;
  name?: string;
  agentId: string;
  status?: string;
  startedAt?: string;
  completedAt?: string;
  latencyMs?: number;
  costUsd?: number;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown> | null;
  tags?: string[];
}

export interface ObservationRecord {
  observationId: string;
  traceId: string;
  parentObservationId?: string | null;
  type?: string;
  name?: string;
  status?: string;
  startedAt?: string | null;
  completedAt?: string | null;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown> | null;
}

export interface GenerationRecord {
  generationId: string;
  traceId: string;
  observationId: string;
  model?: string;
  prompt?: unknown;
  completion?: unknown;
  usage?: LangfuseObservationUsage | null;
  metadata?: Record<string, unknown> | null;
}
