import crypto from 'node:crypto';
import { z } from 'zod';

const SAMPLE_LOCATIONS = [
  'San Francisco',
  'New York',
  'Berlin',
  'Singapore',
  'São Paulo',
  'Bengaluru',
  'Sydney',
];

function randomLocation() {
  return SAMPLE_LOCATIONS[Math.floor(Math.random() * SAMPLE_LOCATIONS.length)];
}

type BuiltinTool = {
  parameters: z.ZodTypeAny;
  execute: (args: Record<string, unknown>) => Promise<unknown> | unknown;
};

const builtinTools: Record<string, BuiltinTool> = {
  getLocation: {
    parameters: z.object({ reason: z.string().optional() }),
    execute: async () => ({ location: randomLocation() }),
  },
  fetchDataset: {
    parameters: z.object({ topic: z.string().optional() }),
    execute: async ({ topic }) => ({
      topic,
      rows: Math.floor(Math.random() * 5000) + 1000,
      source: 'internal-dataset-api',
    }),
  },
  writeMemo: {
    parameters: z.object({ insight: z.string().optional() }),
    execute: async ({ insight }) => ({ id: crypto.randomUUID(), insight }),
  },
};

const defaultTool: BuiltinTool = {
  parameters: z.object({}),
  execute: async () => ({ note: 'tool not implemented' }),
};

export function resolveBuiltinTool(key?: string): BuiltinTool {
  if (!key) {
    return defaultTool;
  }
  return builtinTools[key] ?? defaultTool;
}
