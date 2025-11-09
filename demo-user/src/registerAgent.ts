import { fetch } from 'undici';
import type { ToolRegistrationInput } from '@fledgling/tracer';

const API_BASE = process.env.FLEDGLING_API_BASE ?? 'http://localhost:4000';

interface RegisterAgentPayload {
  id: string;
  name: string;
  taskDescription: string;
  instructions: string;
  originalLLM: string;
  tags?: string[];
  langfuseMetadataKey?: string;
  tools?: ToolRegistrationInput[];
}

export async function registerAgentWithBackend(payload: RegisterAgentPayload): Promise<void> {
  if (!API_BASE) {
    console.warn('FLEDGLING_API_BASE not set; skipping agent registration');
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: payload.id,
        name: payload.name,
        taskDescription: payload.taskDescription,
        instructions: payload.instructions,
        originalLLM: payload.originalLLM,
        tags: payload.tags,
        langfuseMetadataKey: payload.langfuseMetadataKey,
        tools: payload.tools,
      }),
    });

    if (response.ok) {
      console.log(`Registered agent ${payload.id} with backend`);
      return;
    }

    if (response.status === 409) {
      console.log(`Agent ${payload.id} already registered`);
      return;
    }

    const body = await response.text();
    console.error(`Failed to register agent ${payload.id}: ${response.status} ${body}`);
  } catch (error) {
    console.error(`Agent registration error for ${payload.id}:`, error);
  }
}
