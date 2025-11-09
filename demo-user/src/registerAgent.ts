import { fetch } from 'undici';
import type { ToolRegistrationInput } from '@fledgling/tracer';

const API_BASE = process.env.FLEDGLING_API_BASE ?? 'http://localhost:3000';

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
  console.log(`[Agent Registration] Attempting to register agent: ${payload.id} to ${API_BASE}`);
  
  if (!API_BASE) {
    console.warn('FLEDGLING_API_BASE not set; skipping agent registration');
    return;
  }

  try {
    const url = `${API_BASE}/api/agents`;
    console.log(`[Agent Registration] POST ${url}`);
    
    const response = await fetch(url, {
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
      console.log(`✅ Registered agent ${payload.id} with backend`);
      return;
    }

    if (response.status === 409) {
      console.log(`ℹ️  Agent ${payload.id} already registered`);
      return;
    }

    const body = await response.text();
    console.error(`❌ Failed to register agent ${payload.id}: ${response.status} ${body}`);
  } catch (error) {
    console.error(`❌ Agent registration error for ${payload.id}:`, error);
    if (error instanceof Error) {
      console.error(`   Error message: ${error.message}`);
      console.error(`   Error stack: ${error.stack}`);
    }
  }
}
