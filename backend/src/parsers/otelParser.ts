import { LangfuseObservation, LangfuseTrace } from '../service/types';
import {
  ConversationMessage,
  FinetuneSample,
  GenerationStep,
  ParsedStep,
  ThoughtStep,
  ToolCallStep,
  UsageStats,
} from '../types/finetune';

export function parseTraceToSample(trace: LangfuseTrace): FinetuneSample | null {
  const agentId = extractAgentId(trace.metadata);
  if (!agentId) {
    return null;
  }

  const conversation: ConversationMessage[] = [];
  const steps: ParsedStep[] = [];
  let finalResponse: string | undefined;
  let usageTotals: UsageStats | undefined;

  const observations = [...(trace.observations ?? [])].sort((a, b) => {
    const aTime = a.startTime ? Date.parse(a.startTime) : 0;
    const bTime = b.startTime ? Date.parse(b.startTime) : 0;
    return aTime - bTime;
  });

  for (const observation of observations) {
    const parsed = parseObservation(observation);
    if (!parsed) {
      continue;
    }

    if (parsed.conversationMessages?.length) {
      for (const msg of parsed.conversationMessages) {
        // Avoid duplicating identical assistant outputs
        const last = conversation[conversation.length - 1];
        if (!last || last.role !== msg.role || last.content !== msg.content) {
          conversation.push(msg);
        }
      }
    }

    if (parsed.steps?.length) {
      steps.push(...parsed.steps);
    }

    if (parsed.finalResponse && !finalResponse) {
      finalResponse = parsed.finalResponse;
    }

    usageTotals = mergeUsage(usageTotals, parsed.usage);
  }

  if (!conversation.length && !steps.length) {
    return null;
  }

  return {
    traceId: trace.id,
    agentId,
    conversation,
    steps,
    finalResponse,
    usage: usageTotals,
    metadata: trace.metadata ?? null,
  };
}

interface ParsedObservationResult {
  conversationMessages?: ConversationMessage[];
  steps?: ParsedStep[];
  finalResponse?: string;
  usage?: UsageStats;
}

function parseObservation(observation: LangfuseObservation): ParsedObservationResult | null {
  const type = observation.type?.toLowerCase();

  if (type === 'generation') {
    const generationStep: GenerationStep = {
      type: 'generation',
      model: observation.model,
      prompt: observation.input,
      completion: observation.output,
      usage: observation.usage ?? undefined,
      observationId: observation.id,
    };

    return {
      steps: [generationStep],
      finalResponse: stringifyContent(observation.output),
      usage: observation.usage ?? undefined,
    };
  }

  if (type === 'tool') {
    const toolStep: ToolCallStep = {
      type: 'tool_call',
      toolName: observation.name ?? 'tool',
      input: observation.input,
      output: observation.output,
      status: observation.status,
      observationId: observation.id,
      startedAt: observation.startTime ?? undefined,
      completedAt: observation.endTime ?? undefined,
    };

    return { steps: [toolStep] };
  }

  const mastraPayload = extractMastraGenerationPayload(observation);
  if (mastraPayload) {
    return buildResultsFromMastraPayload(observation, mastraPayload);
  }

  // Generic fallback: treat as thought if it has textual output
  const fallbackText = stringifyContent(observation.output ?? observation.input);
  if (fallbackText) {
    const thought: ThoughtStep = {
      type: 'thought',
      content: fallbackText,
      observationId: observation.id,
      timestamp: observation.startTime ?? undefined,
    };

    return { steps: [thought] };
  }

  return null;
}

function buildResultsFromMastraPayload(
  observation: LangfuseObservation,
  payload: MastraGenerationMetadata,
): ParsedObservationResult {
  const conversationMessages = extractConversationMessages(payload);
  const steps: ParsedStep[] = [];

  const thoughtSteps = extractMastraThoughts(payload, observation.id);
  steps.push(...thoughtSteps);

  const toolSteps = extractMastraToolCalls(payload, observation.id);
  steps.push(...toolSteps);

  const completionTextCandidate =
    payload.text ??
    payload.steps?.[payload.steps.length - 1]?.text ??
    normalizeContent(payload.response?.messages?.[0]?.content)
      .map(extractContentText)
      .filter(Boolean)
      .join('\n');
  const completionText = completionTextCandidate && completionTextCandidate.length > 0 ? completionTextCandidate : undefined;

  const generationStep: GenerationStep = {
    type: 'generation',
    model: payload.request?.body?.model,
    prompt: payload.request?.body?.input,
    completion: completionText,
    usage: payload.totalUsage ?? payload.usage ?? undefined,
    observationId: observation.id,
  };

  steps.push(generationStep);

  return {
    conversationMessages,
    steps,
    finalResponse: completionText,
    usage: payload.totalUsage ?? payload.usage ?? undefined,
  };
}

function extractConversationMessages(payload: MastraGenerationMetadata): ConversationMessage[] {
  const inputMessages = Array.isArray(payload.request?.body?.input)
    ? (payload.request?.body?.input as MastraMessage[])
    : [];

  const messages: ConversationMessage[] = [];
  for (const message of inputMessages) {
    if (message.role === 'system' || message.role === 'user') {
      const contentText = normalizeContent(message.content)
        .map(extractContentText)
        .filter(Boolean)
        .join('\n');
      if (contentText) {
        messages.push({ role: message.role, content: contentText });
      }
    }
  }

  if (payload.text) {
    messages.push({ role: 'assistant', content: payload.text });
  }

  return messages;
}

function extractMastraThoughts(payload: MastraGenerationMetadata, observationId?: string): ThoughtStep[] {
  const steps = payload.steps ?? [];
  const thoughts: ThoughtStep[] = [];

  for (const step of steps) {
    const thoughtText =
      step.reasoningText ??
      step.text ??
      normalizeContent(step.content)
        .map(extractContentText)
        .filter(Boolean)
        .join('\n');
    if (thoughtText) {
      thoughts.push({
        type: 'thought',
        content: thoughtText,
        observationId,
      });
    }
  }

  return thoughts;
}

function extractMastraToolCalls(payload: MastraGenerationMetadata, observationId?: string): ToolCallStep[] {
  const steps = payload.steps ?? [];
  const toolSteps: ToolCallStep[] = [];

  for (const step of steps) {
    const candidateLists = [
      (step as unknown as Record<string, unknown>)['toolCalls'],
      (step as unknown as Record<string, unknown>)['toolResults'],
      (step as unknown as Record<string, unknown>)['staticToolCalls'],
      (step as unknown as Record<string, unknown>)['dynamicToolCalls'],
      (step as unknown as Record<string, unknown>)['staticToolResults'],
      (step as unknown as Record<string, unknown>)['dynamicToolResults'],
    ];

    for (const maybeList of candidateLists) {
      if (!Array.isArray(maybeList)) {
        continue;
      }

      for (const entry of maybeList) {
        const record = entry as MastraToolCallLike;
        if (!record) {
          continue;
        }

        const toolStep: ToolCallStep = {
          type: 'tool_call',
          toolName: record.name ?? record.toolName ?? 'tool',
          input: record.args ?? record.input ?? null,
          output: record.output ?? null,
          status: record.status,
          observationId,
        };
        toolSteps.push(toolStep);
      }
    }
  }

  return toolSteps;
}

function mergeUsage(existing?: UsageStats, incoming?: UsageStats): UsageStats | undefined {
  if (!incoming) {
    return existing;
  }
  if (!existing) {
    return { ...incoming };
  }
  return {
    inputTokens: sum(existing.inputTokens, incoming.inputTokens),
    outputTokens: sum(existing.outputTokens, incoming.outputTokens),
    totalTokens: sum(existing.totalTokens, incoming.totalTokens),
    reasoningTokens: sum(existing.reasoningTokens, incoming.reasoningTokens),
    cachedInputTokens: sum(existing.cachedInputTokens, incoming.cachedInputTokens),
  };
}

function sum(a?: number, b?: number): number | undefined {
  if (a === undefined) return b;
  if (b === undefined) return a;
  return a + b;
}

function stringifyContent(value: unknown): string | undefined {
  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map(stringifyContent).filter(Boolean).join('\n') || undefined;
  }

  if (typeof value === 'object' && value !== null) {
    if ('text' in (value as { text?: string })) {
      return ((value as { text?: string }).text as string) ?? undefined;
    }
    return JSON.stringify(value);
  }

  if (value === undefined || value === null) {
    return undefined;
  }

  return String(value);
}

function extractAgentId(metadata?: Record<string, unknown> | null): string | undefined {
  if (!metadata || typeof metadata !== 'object') {
    return undefined;
  }
  const raw = (metadata as Record<string, unknown>)['agent_id'];
  if (typeof raw === 'string' && raw.trim()) {
    return raw;
  }
  return undefined;
}

function safeJsonParse<T>(value: unknown): T | null {
  if (typeof value !== 'string') {
    return null;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

function getMastraAttributes(observation: LangfuseObservation) {
  if (observation.name !== 'agent.generate') {
    return undefined;
  }
  const metadata = observation.metadata;
  if (!metadata || typeof metadata !== 'object' || !('attributes' in metadata)) {
    return undefined;
  }
  const attrs = (metadata as { attributes?: Record<string, unknown> }).attributes;
  if (!attrs || typeof attrs !== 'object') {
    return undefined;
  }
  return attrs as Record<string, unknown>;
}

function extractMastraGenerationPayload(observation: LangfuseObservation): MastraGenerationMetadata | null {
  const attrs = getMastraAttributes(observation);
  if (!attrs) {
    return null;
  }
  return safeJsonParse<MastraGenerationMetadata>(attrs['agent.generate.result']);
}

function extractContentText(part: { type?: string; text?: string } | undefined): string {
  if (!part) {
    return '';
  }
  if (part.type === 'text' && part.text) {
    return part.text;
  }
  if (part.text) {
    return part.text;
  }
  return '';
}

function normalizeContent(
  content: string | Array<{ type?: string; text?: string }> | undefined,
): Array<{ type?: string; text?: string }> {
  if (typeof content === 'string') {
    return [{ type: 'text', text: content }];
  }
  if (Array.isArray(content)) {
    return content;
  }
  return [];
}

interface MastraMessage {
  role: string;
  content?: Array<{ type?: string; text?: string }>;
}

interface MastraToolCallLike {
  name?: string;
  toolName?: string;
  args?: unknown;
  input?: unknown;
  output?: unknown;
  status?: string;
}

interface MastraGenerationMetadata {
  text?: string;
  usage?: UsageStats;
  totalUsage?: UsageStats;
  steps?: Array<{
    text?: string;
    reasoningText?: string;
    content?: Array<{ type?: string; text?: string }>;
    [key: string]: unknown;
  }>;
  request?: {
    body?: {
      input?: unknown;
      model?: string;
    };
  };
  response?: {
    messages?: Array<{ role?: string; content?: Array<{ type?: string; text?: string }> }>;
  };
}
