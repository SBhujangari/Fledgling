import {
  startActiveObservation,
  startObservation,
  updateActiveTrace,
} from '@langfuse/tracing';

import { ensureTracerInitialized } from '../core';

export interface MastraTracingOptions {
  agentId: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export type MastraAgentLike = {
  generate: (...args: any[]) => Promise<any>;
  stream?: (...args: any[]) => Promise<any>;
};

export function withMastraTracing<T extends MastraAgentLike>(agent: T, options: MastraTracingOptions): T {
  ensureTracerInitialized();

  return {
    ...agent,
    generate: instrumentMethod(agent, agent.generate, 'generate', options),
    stream: agent.stream ? instrumentMethod(agent, agent.stream, 'stream', options) : undefined,
  } as T;
}

function instrumentMethod<T extends (...args: any[]) => Promise<any>>(
  agent: MastraAgentLike,
  method: T,
  methodName: string,
  options: MastraTracingOptions,
): T {
  const wrapped = async function instrumented(this: unknown, ...args: Parameters<T>): Promise<ReturnType<T>> {
    return startActiveObservation(
      `mastra-${methodName}`,
      async (agentObservation) => {
        updateActiveTrace({
          metadata: { agent_id: options.agentId, ...(options.metadata ?? {}) },
          tags: options.tags,
        });

        agentObservation.update({
          input: args[0],
          metadata: { agentId: options.agentId },
        });

        const callArgs = attachTracingOptions(
          args,
          agentObservation.traceId,
          agentObservation.otelSpan.spanContext().spanId,
        ) as Parameters<T>;

        try {
          const result = await method.apply(agent, callArgs);
          agentObservation.update({ output: result });
          recordGeneration(agentObservation, args[0], result, options);
          recordToolCalls(agentObservation, result, options);
          return result as ReturnType<T>;
        } catch (error) {
          agentObservation.update({ level: 'ERROR', statusMessage: error instanceof Error ? error.message : String(error) });
          throw error;
        } finally {
          agentObservation.end();
        }
      },
      { asType: 'agent' },
    );
  };

  return wrapped as T;
}

function attachTracingOptions(args: any[], traceId?: string, parentSpanId?: string) {
  const normalized = [...args];
  const tracingOptions = {
    traceId,
    parentSpanId,
  };

  const optionsIndex = typeof normalized[1] === 'object' && normalized[1] !== null ? 1 : -1;
  if (optionsIndex >= 0) {
    normalized[1] = {
      ...normalized[1],
      tracingOptions: {
        ...(normalized[1].tracingOptions ?? {}),
        ...tracingOptions,
      },
    };
  } else {
    normalized[1] = { tracingOptions };
  }

  return normalized;
}

function recordGeneration(agentObservation: any, prompt: unknown, result: any, options: MastraTracingOptions) {
  const completion = result?.text ?? result?.response?.messages?.map(flattenContent).join('\n');
  const usage = mapUsage(result?.usage ?? result?.totalUsage);

  const generation = startObservation(
    'mastra-generation',
    {
      input: prompt,
      output: completion,
      metadata: { agentId: options.agentId },
      model: result?.model,
      usageDetails: usage,
    },
    {
      asType: 'generation',
      parentSpanContext: agentObservation.otelSpan.spanContext(),
    },
  );

  generation.end();
}

function recordToolCalls(agentObservation: any, result: any, options: MastraTracingOptions) {
  const toolCalls: Array<{ name?: string; args?: unknown; result?: unknown; status?: string }> =
    result?.toolCalls ?? result?.raw?.toolCalls ?? [];

  for (const call of toolCalls) {
    const tool = startObservation(
      call.name ?? 'mastra-tool',
      {
        input: call.args,
        output: call.result,
        metadata: { agentId: options.agentId },
        level: call.status === 'error' ? 'ERROR' : undefined,
      },
      {
        asType: 'tool',
        parentSpanContext: agentObservation.otelSpan.spanContext(),
      },
    );
    tool.end();
  }
}

function mapUsage(usage: any): Record<string, number> | undefined {
  if (!usage) {
    return undefined;
  }

  const mapped: Record<string, number> = {};
  if (usage.inputTokens !== undefined) mapped.inputTokens = usage.inputTokens;
  if (usage.outputTokens !== undefined) mapped.outputTokens = usage.outputTokens;
  if (usage.totalTokens !== undefined) mapped.totalTokens = usage.totalTokens;
  if (usage.reasoningTokens !== undefined) mapped.reasoningTokens = usage.reasoningTokens;
  if (usage.cachedInputTokens !== undefined) mapped.cachedInputTokens = usage.cachedInputTokens;
  return Object.keys(mapped).length ? mapped : undefined;
}

function flattenContent(part: any): string {
  if (!part) return '';
  if (typeof part === 'string') return part;
  if (Array.isArray(part)) return part.map(flattenContent).join('\n');
  if (typeof part === 'object' && part !== null) {
    if (typeof part.text === 'string') return part.text;
    if (part.value) return JSON.stringify(part.value);
  }
  return '';
}
