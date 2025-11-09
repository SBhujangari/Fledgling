export interface AgentRecord {
  id: string;
  name: string; // Required: Display name for the agent
  taskDescription: string; // Required: What task this agent is designed to solve
  originalLLM: string; // Required: The LLM model being replaced (e.g., "gpt-4", "claude-3-sonnet")
  slmModel: string; // Auto-assigned: The SLM we're fine-tuning (e.g., "llama-3-8b")
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
  accuracy?: number; // Current model accuracy percentage (0-100)
  modelCostsSaved?: number; // Total cost savings in USD
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
  originalLLM: string; // Required: The LLM model being replaced
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
  accuracy?: number;
  modelCostsSaved?: number;
  // Note: slmModel is NOT in input - it's auto-assigned by the system
}
