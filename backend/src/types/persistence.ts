export interface AgentRecord {
  id: string;
  name: string; // Required: Display name for the agent
  taskDescription: string; // Required: What task this agent is designed to solve
  instructions: string; // Required: System prompt/instructions for the agent
  originalLLM: string; // Required: The LLM model being replaced (e.g., "gpt-4", "claude-3-sonnet")
  slmModel: string; // Auto-assigned: The SLM we're fine-tuning (e.g., "llama-3-8b")
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
  accuracy?: number; // Current model accuracy percentage (0-100)
  modelCostsSaved?: number; // Total cost savings in USD
  toolIds?: string[]; // IDs of tools associated with this agent
  isTraining?: boolean; // Whether the agent is currently being fine-tuned
  iterations?: number; // Number of training iterations completed
  trainingDataSize?: number; // Total number of training samples
  createdAt: string;
  updatedAt: string;
}

export interface AgentStore {
  agents: AgentRecord[];
}

export interface AgentRegistrationInput {
  id: string;
  name: string; // Required
  taskDescription: string; // Required
  instructions: string; // Required: System prompt/instructions for the agent
  originalLLM: string; // Required: The LLM model being replaced
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
  accuracy?: number;
  modelCostsSaved?: number;
  toolIds?: string[];
  iterations?: number;
  trainingDataSize?: number;
  // Note: slmModel is NOT in input - it's auto-assigned by the system
}

export interface ToolRecord {
  id: string;
  name: string;
  description?: string;
  inputSchema?: unknown;
  outputSchema?: unknown;
  metadata?: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface ToolStore {
  tools: ToolRecord[];
}

export interface ToolRegistrationInput {
  id: string;
  name?: string;
  description?: string;
  inputSchema?: unknown;
  outputSchema?: unknown;
  metadata?: Record<string, unknown> | null;
}

export interface RunRecord {
  id: string;
  agentId: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  queuedAt: string;
  completedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface RunStore {
  runs: RunRecord[];
}
