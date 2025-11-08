import { LangfuseClient } from '@langfuse/client';

import { loadLangfuseConfig } from './config';
import type { LangfuseTrace } from './types';

const DEFAULT_FIELDS = 'core,io,observations,metrics';

type StringOperators = '=' | 'contains' | 'does not contain' | 'starts with' | 'ends with';
type DatetimeOperators = '>=' | '>' | '<=' | '<';

type FilterCondition =
  | {
      type: 'stringObject';
      column: 'metadata';
      key: string;
      operator: StringOperators;
      value: string;
    }
  | {
      type: 'string';
      column: string;
      operator: StringOperators;
      value: string;
    }
  | {
      type: 'datetime';
      column: string;
      operator: DatetimeOperators;
      value: string;
    };

export interface FetchOptions {
  updatedAfter?: string; // ISO8601 timestamp supplied by orchestrator
}

function buildFilter(agentId: string, updatedAfter?: string): FilterCondition[] {
  const filter: FilterCondition[] = [
    {
      type: 'stringObject',
      column: 'metadata',
      key: 'agent_id',
      operator: '=',
      value: agentId,
    },
  ];

  if (updatedAfter) {
    filter.push({
      type: 'datetime',
      column: 'updatedAt',
      operator: '>=',
      value: updatedAfter,
    });
  }

  return filter;
}

async function fetchTraceDetail(langfuse: LangfuseClient, traceId: string) {
  const trace = await langfuse.api.trace.get(traceId, {
    queryParams: { fields: DEFAULT_FIELDS },
  });
  return trace;
}

export async function* fetchCompletedTraces(options: FetchOptions = {}) {
  const config = loadLangfuseConfig();
  const langfuse = new LangfuseClient({
    publicKey: config.publicKey,
    secretKey: config.secretKey,
    baseUrl: config.baseUrl,
  });

  const filter = buildFilter(config.agentId, options.updatedAfter);
  let page = 1;

  while (true) {
    const response = await langfuse.api.trace.list({
      limit: config.pageSize,
      page,
      fields: DEFAULT_FIELDS,
      filter: JSON.stringify(filter),
    });

    const traces = response.data ?? [];

    if (!traces.length) {
      break;
    }

    const detailedTraces: LangfuseTrace[] = [];
    for (const trace of traces) {
      try {
        const detail = (await fetchTraceDetail(langfuse, trace.id)) as LangfuseTrace;
        detailedTraces.push(detail);
      } catch (error) {
        console.warn(`Failed to fetch trace ${trace.id}:`, error);
      }
    }

    if (detailedTraces.length) {
      yield detailedTraces;
    }

    const reachedEndByCount = traces.length < config.pageSize;
    const meta = response.meta;
    const reachedEndByMeta =
      meta?.totalPages !== undefined && meta?.page !== undefined && meta.page >= meta.totalPages;

    if (reachedEndByCount || reachedEndByMeta) {
      break;
    }

    page += 1;
  }
}
