export interface ToolCall {
  name: string
  input: Record<string, unknown>
  result: unknown
}

export interface ChatResponse {
  content: string
  toolCalls?: ToolCall[]
}

export interface HistoryItem {
  id: string
  timestamp: string
  iteration: string
  prompt: string
  llmResponse: ChatResponse
  slmResponse: ChatResponse
  slmFallback?: boolean
  traceSample?: TraceSample
}

export interface Iteration {
  id: string
  label: string
  date: string
}

export interface Agent {
  id: string
  name: string
  costSavings: number
  costSavingsPercent: number
  currentAccuracy: number
}

export interface ToolResponse {
  id: string
  name: string
  description?: string
  inputSchema?: unknown
  outputSchema?: unknown
  metadata?: Record<string, unknown> | null
}

export interface AgentResponse {
  id: string
  name: string
  task_description: string
  instructions: string
  original_llm: string
  slm_model: string
  last_updated_at: string
  last_trained_model_path: string | null
  accuracy: number | null
  model_costs_saved: number | null
  tool_ids: string[]
  is_training?: boolean
  iterations?: number
  training_data_size?: number
}

export interface RunResponse {
  id: string
  agentId: string
  name: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  queuedAt: string
  completedAt: string | null
  createdAt: string
  updatedAt: string
}

export interface TracesResponse {
  runs: unknown[]
  observations: unknown[]
  generations: unknown[]
  samples: unknown[]
}

export interface UsageStats {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  reasoningTokens?: number
  cachedInputTokens?: number
}

export interface ThoughtStep {
  type: 'thought'
  content: string
  observationId?: string
  timestamp?: string
}

export interface ToolCallStep {
  type: 'tool_call'
  toolName: string
  input: unknown
  output?: unknown
  status?: string
  observationId?: string
  startedAt?: string
  completedAt?: string
}

export interface GenerationStep {
  type: 'generation'
  model?: string
  prompt?: unknown
  completion?: unknown
  usage?: UsageStats
  observationId?: string
}

export type ParsedStep = ThoughtStep | ToolCallStep | GenerationStep

export interface TraceSample {
  traceId?: string
  agentId?: string
  steps?: ParsedStep[]
  finalResponse?: string
  usage?: UsageStats
}
