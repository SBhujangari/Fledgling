import { LangfuseClient } from '@langfuse/client';

let client: LangfuseClient | null = null;

export function getLangfuseClient(): LangfuseClient {
  if (client) {
    return client;
  }

  const secretKey = process.env.LANGFUSE_SECRET_KEY?.trim() ?? process.env.LANGFUSE_API_KEY?.trim();
  if (!secretKey) {
    throw new Error('Missing LANGFUSE_SECRET_KEY (or LANGFUSE_API_KEY).');
  }

  client = new LangfuseClient({
    secretKey,
    publicKey: process.env.LANGFUSE_PUBLIC_KEY ?? undefined,
    baseUrl: process.env.LANGFUSE_BASE_URL ?? undefined,
  });

  return client;
}
