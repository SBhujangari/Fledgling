import type { AgentResponse, TracesResponse, ToolResponse } from "@/types"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:3000"

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }))
    throw new Error(error.error || `HTTP error! status: ${response.status}`)
  }

  return response.json()
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
}
