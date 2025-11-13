import type {
  AgentResponse,
  ComparisonData,
  ExampleWorkflow,
  HfTokenMeta,
  SlmModelsResponse,
  TracesResponse,
  ToolResponse,
  UploadResult,
} from "@/types"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:4000"

interface UploadRequestPayload {
  repoId: string
  paths: string[]
  commitMessage?: string
  branch?: string
  pathInRepo?: string
  private?: boolean
  autoSubdir?: boolean
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  })

  const rawBody = await response.text()
  let parsedBody: any = rawBody
  if (rawBody) {
    try {
      parsedBody = JSON.parse(rawBody)
    } catch {
      parsedBody = rawBody
    }
  }

  if (!response.ok) {
    const message =
      typeof parsedBody === "object" && parsedBody !== null && "error" in parsedBody
        ? (parsedBody.error as string)
        : typeof parsedBody === "string" && parsedBody.trim().length > 0
          ? parsedBody
          : `HTTP error! status: ${response.status}`
    throw new Error(message)
  }

  return parsedBody as T
}

export const api = {
  // Health check
  health: () => fetchAPI<{ status: string }>("/health"),

  // Agents
  getAgents: () => fetchAPI<AgentResponse[]>("/api/agents"),
  registerAgent: (data: {
    id: string
    name: string
    taskDescription: string
    instructions: string
    originalLLM: string
    tags?: string[]
    langfuseMetadataKey?: string
    lastTrainedModelPath?: string
  }) =>
    fetchAPI<AgentResponse>("/api/agents", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Traces
  getTraces: (params?: { updatedAfter?: string }) => {
    const queryParams = params?.updatedAfter
      ? `?updatedAfter=${encodeURIComponent(params.updatedAfter)}`
      : ""
    return fetchAPI<TracesResponse>(`/api/traces${queryParams}`)
  },

  // Local traces (self-hosted, no Langfuse key needed)
  getLocalTraces: () => fetchAPI<{ traces: any[] }>("/api/traces/local"),

  getLocalTraceById: (traceId: string) => fetchAPI<any>(`/api/traces/local/${traceId}`),

  train: () =>
    fetchAPI<{ status: string }>("/api/train", {
      method: "POST",
    }),

  // LangFuse connection (for onboarding)
  connectLangFuse: (data: {
    publicKey: string
    secretKey: string
    baseUrl: string
  }) =>
    fetchAPI<{ success: boolean; message: string; data?: unknown }>(
      "/api/langfuse/connect",
      {
        method: "POST",
        body: JSON.stringify(data),
      }
    ),

  // Tools
  getTools: (agentId?: string) => {
    const queryParams = agentId ? `?agentId=${encodeURIComponent(agentId)}` : ""
    return fetchAPI<ToolResponse[]>(`/api/tools${queryParams}`)
  },

  getExampleWorkflows: () => fetchAPI<ExampleWorkflow[]>("/api/examples/workflows"),

  // Hugging Face + SLM management
  uploadHfArtifacts: async (payload: UploadRequestPayload) => {
    const response = await fetchAPI<{ ok: boolean; result: UploadResult }>("/api/hf/upload", {
      method: "POST",
      body: JSON.stringify(payload),
    })
    return response.result
  },
  getHfTokenMeta: () => fetchAPI<HfTokenMeta>("/api/hf/token"),
  saveHfToken: (token: string) =>
    fetchAPI<{ ok: boolean; updatedAt?: string }>("/api/hf/token", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
  clearHfToken: () =>
    fetchAPI<{ ok: boolean }>("/api/hf/token", {
      method: "DELETE",
    }),
  getSlmCatalog: () => fetchAPI<SlmModelsResponse>("/api/slm/models"),
  selectSlmModel: async (modelId: string) => {
    const response = await fetchAPI<{ ok: boolean; selection: { modelId: string; selectedAt: string } }>("/api/slm/select", {
      method: "POST",
      body: JSON.stringify({ modelId }),
    })
    return response.selection
  },

  // Metrics + comparisons
  getMetricsComparison: () => fetchAPI<ComparisonData>("/api/metrics/comparison"),
}
