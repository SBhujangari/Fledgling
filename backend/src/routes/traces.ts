import { Router, Request, Response } from 'express';

import { fetchCompletedTraces } from '../service/loader';
import { transformTraces, type TransformResult } from '../service/transformer';
import { parseTraceToSample } from '../parsers/otelParser';
import { ensureAgentRegistered } from '../service/store/agentStore';
import type { FinetuneSample } from '../types/finetune';

const router = Router();

router.get('/traces', async (req: Request, res: Response) => {
  try {
    const updatedAfter = typeof req.query.updatedAfter === 'string' ? req.query.updatedAfter : undefined;
    const aggregate: TransformResult = {
      runs: [],
      observations: [],
      generations: [],
    };
    const samples: FinetuneSample[] = [];

    for await (const page of fetchCompletedTraces({ updatedAfter })) {
      const transformed = transformTraces(page);
      aggregate.runs.push(...transformed.runs);
      aggregate.observations.push(...transformed.observations);
      aggregate.generations.push(...transformed.generations);

      for (const trace of page) {
        const sample = parseTraceToSample(trace);
        if (sample) {
          const metadataAgentName =
            sample.metadata && typeof sample.metadata === 'object'
              ? (sample.metadata['agent_name'] as string | undefined)
              : undefined;

          await ensureAgentRegistered(sample.agentId, metadataAgentName ?? trace.name ?? sample.agentId);
          samples.push(sample);
        }
      }
    }

    res.json({ ...aggregate, samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

router.post('/train', (_req: Request, res: Response) => {
  res.json({ status: 'coming soon' });
});

export default router;
