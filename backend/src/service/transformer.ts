import {
  AgentRun,
  GenerationRecord,
  LangfuseObservation,
  LangfuseTrace,
  ObservationRecord,
} from './types';

export interface TransformResult {
  runs: AgentRun[];
  observations: ObservationRecord[];
  generations: GenerationRecord[];
}

function assertAgentId(metadata?: Record<string, unknown> | null): string {
  if (metadata && typeof metadata === 'object') {
    const raw = (metadata as Record<string, unknown>)['agent_id'];
    if (typeof raw === 'string' && raw.trim().length > 0) {
      return raw;
    }
  }

  throw new Error('Trace missing required metadata.agent_id. Ensure integration attaches agent identifiers.');
}

function transformAgentRun(trace: LangfuseTrace): AgentRun {
  const agentId = assertAgentId(trace.metadata);
  const startedAt = trace.startTime ?? trace.timestamp ?? trace.createdAt;
  const completedAt = trace.endTime ?? trace.updatedAt;
  return {
    traceId: trace.id,
    name: trace.name,
    agentId,
    status: trace.status,
    startedAt,
    completedAt,
    latencyMs: trace.latency,
    costUsd: trace.totalCost,
    input: trace.input,
    output: trace.output,
    metadata: trace.metadata ?? null,
    tags: trace.tags,
  };
}

function transformObservation(traceId: string, observation: LangfuseObservation): ObservationRecord | null {
  if (!observation.id) {
    return null;
  }

  return {
    observationId: observation.id,
    traceId,
    parentObservationId: observation.parentObservationId ?? null,
    type: observation.type,
    name: observation.name,
    status: observation.status,
    startedAt: observation.startTime ?? null,
    completedAt: observation.endTime ?? null,
    input: observation.input,
    output: observation.output,
    metadata: observation.metadata ?? null,
  };
}

function transformGeneration(traceId: string, observation: LangfuseObservation): GenerationRecord | null {
  const isGeneration = observation.type?.toLowerCase() === 'generation';
  if (!isGeneration || !observation.id) {
    return null;
  }

  return {
    generationId: observation.id,
    traceId,
    observationId: observation.id,
    model: observation.model,
    prompt: observation.input,
    completion: observation.output,
    usage: observation.usage ?? null,
    metadata: observation.metadata ?? null,
  };
}

export function transformTraces(traces: LangfuseTrace[]): TransformResult {
  const runs: AgentRun[] = [];
  const observations: ObservationRecord[] = [];
  const generations: GenerationRecord[] = [];

  for (const trace of traces) {
    runs.push(transformAgentRun(trace));

    const obsList = trace.observations ?? [];
    for (const obs of obsList) {
      const normalizedObs = transformObservation(trace.id, obs);
      if (normalizedObs) {
        observations.push(normalizedObs);
      }

      const generation = transformGeneration(trace.id, obs);
      if (generation) {
        generations.push(generation);
      }
    }
  }

  return { runs, observations, generations };
}
