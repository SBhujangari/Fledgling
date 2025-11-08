import express from 'express';
import cors from 'cors';

import { fetchCompletedTraces } from './service/loader';
import { transformTraces, type TransformResult } from './service/transformer';

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.get('/api/traces', async (req, res) => {
  try {
    const updatedAfter = typeof req.query.updatedAfter === 'string' ? req.query.updatedAfter : undefined;
    const aggregate: TransformResult = {
      runs: [],
      observations: [],
      generations: [],
    };

    for await (const page of fetchCompletedTraces({ updatedAfter })) {
      const transformed = transformTraces(page);
      aggregate.runs.push(...transformed.runs);
      aggregate.observations.push(...transformed.observations);
      aggregate.generations.push(...transformed.generations);
    }

    res.json(aggregate);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

const PORT = Number(process.env.PORT) || 4000;
app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});
