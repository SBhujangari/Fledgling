import type { LangfuseTrace } from './types';
import { parseTraceToSample } from '../parsers/otelParser';
import type { FinetuneSample } from '../types/finetune';
import { getLangfuseClient } from './langfuseClient';

const TRACE_FIELDS = 'core,io,observations,metrics';

export async function fetchLatestTraceSample(agentId: string): Promise<FinetuneSample | null> {
  try {
    const client = getLangfuseClient();

    const filter = JSON.stringify([
      {
        type: 'stringObject',
        column: 'metadata',
        key: 'agent_id',
        operator: '=',
        value: agentId,
      },
    ]);

    const list = await client.api.trace.list({
      limit: 1,
      orderBy: 'timestamp.desc',
      filter,
      fields: TRACE_FIELDS,
    });

    const trace = list.data?.[0];
    if (!trace) {
      return null;
    }

    const detail = (await client.api.trace.get(trace.id, {
      queryParams: { fields: TRACE_FIELDS },
    })) as LangfuseTrace;

    return parseTraceToSample(detail);
  } catch (error) {
    console.error('Failed to fetch latest trace sample:', error);
    return null;
  }
}
