import fs from 'fs/promises';
import path from 'path';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';

import type { AgentRecord, AgentRegistrationInput, AgentStore } from '../../types/persistence';
import { selectSLMForAgent } from '../../utils/modelSelector';

const STORAGE_DIR = process.env.STORAGE_DIR ?? path.resolve(__dirname, '../../../storage');
const AGENTS_FILE = path.join(STORAGE_DIR, 'agents.json');

let dbPromise: Promise<Low<AgentStore>> | null = null;

async function initDb(): Promise<Low<AgentStore>> {
  if (!dbPromise) {
    dbPromise = (async () => {
      await fs.mkdir(STORAGE_DIR, { recursive: true });
      const adapter = new JSONFile<AgentStore>(AGENTS_FILE);
      const db = new Low<AgentStore>(adapter, { agents: [] });
      await db.read();
      db.data ||= { agents: [] };
      return db;
    })();
  }

  return dbPromise;
}

export async function listAgents(): Promise<AgentRecord[]> {
  const db = await initDb();
  return db.data?.agents ?? [];
}

export async function getAgentById(id: string): Promise<AgentRecord | undefined> {
  const db = await initDb();
  return db.data?.agents.find((agent) => agent.id === id);
}

export async function registerAgent(input: AgentRegistrationInput): Promise<AgentRecord> {
  const db = await initDb();

  if (!input.id || !input.id.trim()) {
    throw new Error('Agent id is required');
  }

  if (!input.name || !input.name.trim()) {
    throw new Error('Agent name is required');
  }

  if (!input.taskDescription || !input.taskDescription.trim()) {
    throw new Error('Agent taskDescription is required');
  }

  if (!input.originalLLM || !input.originalLLM.trim()) {
    throw new Error('Agent originalLLM is required');
  }

  if (!input.instructions || !input.instructions.trim()) {
    throw new Error('Agent instructions is required');
  }

  const existing = db.data?.agents.find((agent) => agent.id === input.id);
  if (existing) {
    throw new Error(`Agent id "${input.id}" is already registered`);
  }

  // Auto-select the best SLM for this agent
  const slmModel = selectSLMForAgent(input.taskDescription, input.originalLLM);

  const timestamp = new Date().toISOString();
  const record: AgentRecord = {
    id: input.id.trim(),
    name: input.name.trim(),
    taskDescription: input.taskDescription.trim(),
    instructions: input.instructions.trim(),
    originalLLM: input.originalLLM.trim(),
    slmModel,
    tags: input.tags,
    langfuseMetadataKey: input.langfuseMetadataKey,
    lastTrainedModelPath: input.lastTrainedModelPath ?? null,
    accuracy: input.accuracy,
    modelCostsSaved: input.modelCostsSaved,
    toolIds: input.toolIds ?? [],
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  db.data?.agents.push(record);
  await db.write();

  return record;
}

/**
 * @deprecated No longer auto-registers agents. Use registerAgent() explicitly.
 * This function now only returns existing agents or throws an error.
 */
export async function ensureAgentRegistered(id: string, name?: string): Promise<AgentRecord | null> {
  const existing = await getAgentById(id);
  return existing || null;
}

export async function updateAgentMetrics(
  id: string,
  metrics: { accuracy?: number; modelCostsSaved?: number }
): Promise<AgentRecord> {
  const db = await initDb();
  const agent = db.data?.agents.find((a) => a.id === id);

  if (!agent) {
    throw new Error(`Agent id "${id}" not found`);
  }

  if (metrics.accuracy !== undefined) {
    agent.accuracy = metrics.accuracy;
  }
  if (metrics.modelCostsSaved !== undefined) {
    agent.modelCostsSaved = metrics.modelCostsSaved;
  }
  agent.updatedAt = new Date().toISOString();

  await db.write();
  return agent;
}
