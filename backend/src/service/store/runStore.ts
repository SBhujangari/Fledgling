import fs from 'fs/promises';
import path from 'path';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';

import type { RunRecord, RunStore } from '../../types/persistence';

const STORAGE_DIR = process.env.STORAGE_DIR ?? path.resolve(__dirname, '../../../storage');
const RUNS_FILE = path.join(STORAGE_DIR, 'runs.json');

let dbPromise: Promise<Low<RunStore>> | null = null;

async function initDb(): Promise<Low<RunStore>> {
  if (!dbPromise) {
    dbPromise = (async () => {
      await fs.mkdir(STORAGE_DIR, { recursive: true });
      const adapter = new JSONFile<RunStore>(RUNS_FILE);
      const db = new Low<RunStore>(adapter, { runs: [] });
      await db.read();
      db.data ||= { runs: [] };
      return db;
    })();
  }

  return dbPromise;
}

export async function listRuns(agentId?: string): Promise<RunRecord[]> {
  const db = await initDb();
  const runs = db.data?.runs ?? [];
  if (agentId) {
    return runs.filter((run) => run.agentId === agentId);
  }
  return runs;
}

export async function getRunById(id: string): Promise<RunRecord | undefined> {
  const db = await initDb();
  return db.data?.runs.find((run) => run.id === id);
}

export async function createRun(agentId: string, name: string): Promise<RunRecord> {
  const db = await initDb();

  if (!agentId || !agentId.trim()) {
    throw new Error('Agent id is required');
  }

  if (!name || !name.trim()) {
    throw new Error('Run name is required');
  }

  const timestamp = new Date().toISOString();
  const runId = `${agentId}-${Date.now()}`;
  
  const record: RunRecord = {
    id: runId,
    agentId: agentId.trim(),
    name: name.trim(),
    status: 'queued',
    queuedAt: timestamp,
    completedAt: null,
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  db.data?.runs.push(record);
  await db.write();

  return record;
}

export async function updateRunStatus(
  id: string,
  status: RunRecord['status'],
  completedAt?: string | null
): Promise<RunRecord> {
  const db = await initDb();
  const run = db.data?.runs.find((r) => r.id === id);

  if (!run) {
    throw new Error(`Run id "${id}" not found`);
  }

  run.status = status;
  run.updatedAt = new Date().toISOString();
  
  if (completedAt !== undefined) {
    run.completedAt = completedAt;
  } else if (status === 'completed' || status === 'failed') {
    run.completedAt = new Date().toISOString();
  }

  await db.write();
  return run;
}

