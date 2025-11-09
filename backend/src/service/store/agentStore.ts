import fs from 'fs/promises';
import path from 'path';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';

import type { AgentRecord, AgentRegistrationInput, AgentStore } from '../../types/persistence';

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
  if (!input.id) {
    throw new Error('Agent id is required');
  }

  const existing = db.data?.agents.find((agent) => agent.id === input.id);
  if (existing) {
    throw new Error(`Agent id "${input.id}" is already registered`);
  }

  const timestamp = new Date().toISOString();
  const record: AgentRecord = {
    id: input.id,
    name: input.name?.trim() || input.id,
    description: input.description,
    tags: input.tags,
    langfuseMetadataKey: input.langfuseMetadataKey,
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  db.data?.agents.push(record);
  await db.write();

  return record;
}

export async function ensureAgentRegistered(id: string, name?: string): Promise<AgentRecord> {
  const existing = await getAgentById(id);
  if (existing) {
    return existing;
  }

  return registerAgent({ id, name });
}
