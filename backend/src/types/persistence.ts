export interface AgentRecord {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface AgentStore {
  agents: AgentRecord[];
}

export interface AgentRegistrationInput {
  id: string;
  name?: string;
  description?: string;
  tags?: string[];
  langfuseMetadataKey?: string;
  lastTrainedModelPath?: string | null;
}
