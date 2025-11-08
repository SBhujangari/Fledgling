import fs from 'fs';
import path from 'path';
import { config as loadEnv } from 'dotenv';

const DEFAULT_BASE_URL = 'https://cloud.langfuse.com';
const DEFAULT_PAGE_SIZE = 50;
const TRACE_STATUS = 'completed' as const;

const defaultEnvPath = path.resolve(__dirname, '..', '..', '.env');
const customEnvPath = process.env.LANGFUSE_ENV_FILE;
const envPathToUse = customEnvPath ?? defaultEnvPath;

if (fs.existsSync(envPathToUse)) {
  loadEnv({ path: envPathToUse });
}

export interface LangfuseConfig {
  secretKey: string;
  publicKey?: string;
  baseUrl: string;
  agentId: string;
  pageSize: number;
  status: typeof TRACE_STATUS;
}

export function loadLangfuseConfig(): LangfuseConfig {
  const secretKey = process.env.LANGFUSE_SECRET_KEY?.trim() ?? process.env.LANGFUSE_API_KEY?.trim();
  if (!secretKey) {
    throw new Error('Missing LANGFUSE_SECRET_KEY (or LANGFUSE_API_KEY). Set it in backend/.env or environment variables.');
  }

  const publicKey = process.env.LANGFUSE_PUBLIC_KEY?.trim();

  const agentId = process.env.LANGFUSE_AGENT_ID?.trim();
  if (!agentId) {
    throw new Error('Missing LANGFUSE_AGENT_ID. Each trace must include a stable agent identifier.');
  }

  const baseUrl = process.env.LANGFUSE_BASE_URL?.trim() || DEFAULT_BASE_URL;

  return {
    secretKey,
    publicKey,
    agentId,
    baseUrl,
    pageSize: DEFAULT_PAGE_SIZE,
    status: TRACE_STATUS,
  };
}
