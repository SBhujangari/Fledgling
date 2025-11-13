import { Agent } from '@mastra/core/agent';
import { z } from 'zod';

import { withMastraTracing } from '../../tracer/adapters/mastra';
import { ensureTracerInitialized } from '../../tracer/core';
import type { ToolRecord } from '../../types/persistence';
import { resolveBuiltinTool } from '../tools/builtins';
import { AgentComparisonAdapter } from './adapters';
import { AgentExecutionPayload, AgentExecutionResult, ModelEndpointConfig } from './types';

export class MastraComparisonAdapter implements AgentComparisonAdapter {
  name = 'mastra';

  supports(config: ModelEndpointConfig): boolean {
    return config.provider === 'mastra';
  }

  async run(config: ModelEndpointConfig, payload: AgentExecutionPayload): Promise<AgentExecutionResult> {
    ensureTracerInitialized();

    const tools = buildMastraTools(payload.tools);
    const agent = new Agent({
      name: payload.agentName,
      instructions: payload.instructions,
      model: config.model as any,
      tools,
    });

    const tracedAgent = withMastraTracing(agent, {
      agentId: payload.agentId,
      tags: ['comparison'],
    });

    const startedAt = Date.now();
    try {
      const result = await tracedAgent.generate(payload.prompt);
      return {
        output: result?.text ?? '',
        latencyMs: Date.now() - startedAt,
      };
    } catch (error) {
      console.error('[MastraAdapter] Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        error,
      });
      return {
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}

function buildMastraTools(
  toolRecords: ToolRecord[],
): Record<string, { description?: string; parameters: z.ZodTypeAny; execute: (args: Record<string, unknown>) => Promise<unknown> | unknown }> {
  const tools: Record<string, { description?: string; parameters: z.ZodTypeAny; execute: (args: Record<string, unknown>) => Promise<unknown> | unknown }> = {};

  for (const record of toolRecords) {
    const metadata = (record.metadata ?? {}) as Record<string, unknown>;
    const builtinKey = typeof metadata?.builtinKey === 'string' ? metadata.builtinKey : record.name;
    const builtin = resolveBuiltinTool(builtinKey);

    tools[record.name ?? record.id] = {
      description: record.description,
      parameters: builtin.parameters ?? z.object({}),
      execute: builtin.execute,
    };
  }

  return tools;
}
