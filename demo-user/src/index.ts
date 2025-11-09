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
    console.error('Customer service agent execution error:', error);
    res.status(500).json({
      error: 'Failed to execute customer service agent',
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Customer Service Agent server running on http://localhost:${PORT}`);
  console.log(`📝 POST /query with { "task": "customer service request" }`);
  console.log(`💡 Example: { "task": "I need to return order #12345, the item was defective" }`);
});
