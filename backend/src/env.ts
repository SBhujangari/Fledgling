import fs from 'fs';
import path from 'path';
import { config as loadEnv } from 'dotenv';

const BACKEND_ROOT = path.resolve(__dirname, '..');
const defaultEnvPath = path.join(BACKEND_ROOT, '.env');
const customEnvPath = process.env.BACKEND_ENV_FILE ?? process.env.LANGFUSE_ENV_FILE;
const envPathToUse = customEnvPath ?? defaultEnvPath;

if (fs.existsSync(envPathToUse)) {
  loadEnv({ path: envPathToUse });
}
