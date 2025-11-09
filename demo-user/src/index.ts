import 'dotenv/config';
import express from 'express';

import { qaAgent } from './agent';
import { researchAgent } from './researchAgent';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());

app.post('/query', async (req, res) => {
  const { task } = req.body;

  if (!task || typeof task !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid "task" field' });
  }

  try {
    const { text } = await qaAgent.generate(task, {
      providerOptions: {
        openai: {
          reasoningEffort: 'high',
          reasoningSummary: 'auto'
        },
      },
    });

    res.json({
      task,
      response: text,
    });
  } catch (error) {
    console.error('Agent execution error:', error);
    res.status(500).json({
      error: 'Failed to execute agent',
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post('/research', async (req, res) => {
  const brief = typeof req.body?.brief === 'string' ? req.body.brief : req.body?.prompt;

  if (!brief || typeof brief !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid "brief" field' });
  }

  try {
    const { text } = await researchAgent.generate(brief, {
      providerOptions: {
        openai: {
          reasoningEffort: 'medium',
          reasoningSummary: 'auto',
        },
      },
    });

    res.json({
      brief,
      response: text,
    });
  } catch (error) {
    console.error('Research agent execution error:', error);
    res.status(500).json({
      error: 'Failed to execute research agent',
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Demo server running on http://localhost:${PORT}`);
  console.log(`📝 POST /query with { "task": "your question" }`);
});
