import express from 'express';
import cors from 'cors';

import agentsRouter from './routes/agents';
import tracesRouter from './routes/traces';
import queryRouter from './routes/query';

const app = express();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.use('/api/agents', agentsRouter);
app.use('/api', tracesRouter);
app.use(queryRouter);

export { app };
export default app;
