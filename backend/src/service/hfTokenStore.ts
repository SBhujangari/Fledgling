import fs from 'fs';
import path from 'path';

const BACKEND_ROOT = path.resolve(__dirname, '..', '..');
const TOKEN_PATH = path.join(BACKEND_ROOT, '.hf_token');

interface TokenFile {
  token: string;
  updatedAt: string;
}

function readTokenFile(): TokenFile | null {
  if (!fs.existsSync(TOKEN_PATH)) {
    return null;
  }
  try {
    const raw = fs.readFileSync(TOKEN_PATH, 'utf-8');
    const data = JSON.parse(raw) as TokenFile;
    if (typeof data.token === 'string' && typeof data.updatedAt === 'string') {
      return data;
    }
    return null;
  } catch {
    return null;
  }
}

export function readStoredHfToken(): string | null {
  const data = readTokenFile();
  return data?.token ?? null;
}

export function writeStoredHfToken(token: string): { updatedAt: string } {
  if (!token || !token.startsWith('hf_')) {
    throw new Error('Token must start with hf_.');
  }
  const payload: TokenFile = {
    token,
    updatedAt: new Date().toISOString(),
  };
  fs.writeFileSync(TOKEN_PATH, JSON.stringify(payload, null, 2), { mode: 0o600 });
  return { updatedAt: payload.updatedAt };
}

export function deleteStoredHfToken(): void {
  if (fs.existsSync(TOKEN_PATH)) {
    fs.rmSync(TOKEN_PATH);
  }
}

export function hasStoredHfToken(): boolean {
  return Boolean(readTokenFile());
}

export function getStoredTokenMetadata(): { hasToken: boolean; updatedAt: string | null } {
  const data = readTokenFile();
  return {
    hasToken: Boolean(data?.token),
    updatedAt: data?.updatedAt ?? null,
  };
}
