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

export interface AgentRun {
  traceId: string
  name?: string
  agentId: string
  status?: string
  startedAt?: string | null
  completedAt?: string | null
  latencyMs?: number | null
  costUsd?: number | null
}

export interface ObservationRecord {
  observationId: string
  traceId: string
  parentObservationId?: string | null
  type?: string | null
  name?: string | null
  status?: string | null
  startedAt?: string | null
  completedAt?: string | null
  metadata?: Record<string, unknown> | null
}

export interface GenerationRecord {
  generationId: string
  traceId: string
  observationId: string
  model?: string | null
  prompt?: unknown
  completion?: unknown
  usage?: {
    inputTokens?: number
    outputTokens?: number
    totalTokens?: number
  } | null
  metadata?: Record<string, unknown> | null
}

export interface ConversationMessage {
  role: "system" | "user" | "assistant"
  content: string
}

export interface FinetuneSample {
  traceId: string
  agentId: string
  conversation: ConversationMessage[]
  steps: Array<{
    type: "thought" | "tool_call" | "generation"
    content?: string
    toolName?: string
    status?: string
  }>
  finalResponse?: string
}

export interface TracesResponse {
  runs: AgentRun[]
  observations: ObservationRecord[]
  generations: GenerationRecord[]
  samples: FinetuneSample[]
}

export interface ExampleWorkflow {
  id: string
  title: string
  summary: string
  datasetPath: string
  evaluationCommand: string
  features: string[]
  codeSnippet: string
  defaultAgent: {
    id: string
    name: string
    taskDescription: string
    instructions: string
    originalLLM: string
    tags: string[]
  }
}

export interface UploadResult {
  repoId: string
  repoType: string
  private: boolean
  paths: string[]
  command: string[]
  stdout: string
  stderr: string
  durationMs: number
}

export interface SlmModel {
  id: string
  label: string
  source: string
  description?: string
  capabilities?: string[]
  available: boolean
  location: string | null
}

export interface SlmModelsResponse {
  models: SlmModel[]
  selectedModelId: string | null
  selectedAt: string | null
}

export interface HfTokenMeta {
  hasToken: boolean
  updatedAt: string | null
}

export interface MetricTrackStats {
  json_valid_rate: number
  field_f1: number
  field_precision?: number
  field_recall?: number
}

export interface MetricDelta {
  f1_pct_of_azure: number
  f1_delta: number
}

export interface ComparisonTrack {
  slm?: MetricTrackStats | null
  azure?: MetricTrackStats | null
  delta?: MetricDelta | null
}

export interface ComparisonSummary {
  message: string
  recommendation: string
}

export interface ComparisonData {
  structured: ComparisonTrack
  toolcall: ComparisonTrack
  summary: ComparisonSummary
}
