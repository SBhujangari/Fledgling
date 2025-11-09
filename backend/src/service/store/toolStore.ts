import fs from 'fs/promises';
import path from 'path';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';

import type { ToolRecord, ToolRegistrationInput, ToolStore } from '../../types/persistence';

const STORAGE_DIR = process.env.STORAGE_DIR ?? path.resolve(__dirname, '../../../storage');
const TOOLS_FILE = path.join(STORAGE_DIR, 'tools.json');

let dbPromise: Promise<Low<ToolStore>> | null = null;

async function initDb(): Promise<Low<ToolStore>> {
  if (!dbPromise) {
    dbPromise = (async () => {
      await fs.mkdir(STORAGE_DIR, { recursive: true });
      const adapter = new JSONFile<ToolStore>(TOOLS_FILE);
      const db = new Low<ToolStore>(adapter, { tools: [] });
      await db.read();
      db.data ||= { tools: [] };
      return db;
    })();
  }

  return dbPromise;
}

export async function listAllTools(): Promise<ToolRecord[]> {
  const db = await initDb();
  return db.data?.tools ?? [];
}

export async function listToolsByIds(ids: string[]): Promise<ToolRecord[]> {
  if (!ids.length) return [];
  const db = await initDb();
  const wanted = new Set(ids);
  return (db.data?.tools ?? []).filter((tool) => wanted.has(tool.id));
}

export async function getToolById(id: string): Promise<ToolRecord | undefined> {
  const db = await initDb();
  return db.data?.tools.find((tool) => tool.id === id);
}

export async function registerOrUpdateTool(input: ToolRegistrationInput): Promise<ToolRecord> {
  if (!input.id || !input.id.trim()) {
    throw new Error('Tool id is required');
  }

  const db = await initDb();
  const normalizedId = input.id.trim();
  const timestamp = new Date().toISOString();
  const existing = db.data?.tools.find((tool) => tool.id === normalizedId);

  if (existing) {
    existing.name = input.name?.trim() || existing.name;
    existing.description = input.description ?? existing.description;
    existing.inputSchema = input.inputSchema ?? existing.inputSchema;
    existing.outputSchema = input.outputSchema ?? existing.outputSchema;
    existing.metadata = input.metadata ?? existing.metadata ?? null;
    existing.updatedAt = timestamp;
    await db.write();
    return existing;
  }

  const record: ToolRecord = {
    id: normalizedId,
    name: input.name?.trim() || normalizedId,
    description: input.description,
    inputSchema: input.inputSchema,
    outputSchema: input.outputSchema,
    metadata: input.metadata ?? null,
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  db.data?.tools.push(record);
  await db.write();
  return record;
}

export async function registerTools(inputs: ToolRegistrationInput[] | undefined): Promise<ToolRecord[]> {
  if (!inputs?.length) {
    return [];
  }

  const results: ToolRecord[] = [];
  for (const tool of inputs) {
    const record = await registerOrUpdateTool(tool);
    results.push(record);
  }
  return results;
}
