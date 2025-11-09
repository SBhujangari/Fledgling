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
}

export interface TracesResponse {
  runs: unknown[]
  observations: unknown[]
  generations: unknown[]
  samples: unknown[]
}
