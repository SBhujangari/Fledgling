import express from 'express';
import cors from 'cors';

import agentsRouter from './routes/agents';
import tracesRouter from './routes/traces';
import toolsRouter from './routes/tools';
import compareRouter from './routes/compare';
import modelsRouter from './routes/models';
import queryRouter from './routes/query';
import trainingRouter from './routes/training';
import examplesRouter from './routes/examples';
import metricsRouter from './routes/metrics';
import hfRouter from './routes/hf';
import slmRouter from './routes/slm';
import judgeRouter from './routes/judge';
import { initTracer } from './tracer/core';

const app = express();

const langfuseSecret = process.env.LANGFUSE_SECRET_KEY ?? process.env.LANGFUSE_API_KEY;
if (langfuseSecret) {
  initTracer({
    secretKey: langfuseSecret,
    publicKey: process.env.LANGFUSE_PUBLIC_KEY ?? undefined,
    baseUrl: process.env.LANGFUSE_BASE_URL ?? undefined,
    environment: process.env.NODE_ENV,
  });
}

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.use('/api/agents', agentsRouter);
app.use('/api', tracesRouter);
app.use('/api/tools', toolsRouter);
app.use('/api/compare', compareRouter);
app.use('/api/models', modelsRouter);
app.use('/api/training', trainingRouter);
app.use('/api/examples', examplesRouter);
app.use('/api/metrics', metricsRouter);
app.use('/api/hf', hfRouter);
app.use('/api/slm', slmRouter);
app.use('/api/judge', judgeRouter);
app.use(queryRouter);

export { app };
export default app;
