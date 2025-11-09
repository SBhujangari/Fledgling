import { Router, Request, Response } from 'express';

import { compareAgentModels } from '../service/comparison';

const router = Router();

router.post('/', async (req: Request, res: Response) => {
  const { agentId, prompt, llmModel } = req.body ?? {};

  if (typeof agentId !== 'string' || !agentId.trim()) {
    return res.status(400).json({ error: 'agentId is required' });
  }

  if (typeof prompt !== 'string' || !prompt.trim()) {
    return res.status(400).json({ error: 'prompt is required' });
  }

  try {
    const normalizedModel = typeof llmModel === 'string'
      ? { provider: 'mastra', model: llmModel }
      : llmModel;

    const result = await compareAgentModels({
      agentId: agentId.trim(),
      prompt: prompt.trim(),
      llmModel: normalizedModel,
    });

    res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

export default router;
