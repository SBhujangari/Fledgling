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

export interface AgentResponse {
  id: string
  last_updated_at: string
  last_trained_model_path: string | null
}

export interface TracesResponse {
  runs: unknown[]
  observations: unknown[]
  generations: unknown[]
  samples: unknown[]
}

