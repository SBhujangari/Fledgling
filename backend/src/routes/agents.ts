import { Router, Request, Response } from 'express';

import { listAgents, registerAgent } from '../service/store/agentStore';

const router = Router();

router.get('/', async (_req: Request, res: Response) => {
  const agents = await listAgents();
  const payload = agents.map((agent) => ({
    id: agent.id,
    last_updated_at: agent.updatedAt,
    last_trained_model_path: agent.lastTrainedModelPath ?? null,
  }));

  res.json(payload);
});

router.post('/', async (req: Request, res: Response) => {
  const { id, name, description, tags, langfuseMetadataKey, lastTrainedModelPath } = req.body ?? {};

  if (typeof id !== 'string' || id.trim().length === 0) {
    return res.status(400).json({ error: 'id is required' });
  }

  try {
    const agent = await registerAgent({
      id: id.trim(),
      name,
      description,
      tags,
      langfuseMetadataKey,
      lastTrainedModelPath,
    });

    res.status(201).json(agent);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    const status = message.includes('already registered') ? 409 : 500;
    res.status(status).json({ error: message });
  }
});

export default router;
