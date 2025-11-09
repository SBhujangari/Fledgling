import { Router, Request, Response } from 'express';

import { listAgents, registerAgent } from '../service/store/agentStore';

const router = Router();

router.get('/', async (_req: Request, res: Response) => {
  const agents = await listAgents();
  const payload = agents.map((agent) => ({
    id: agent.id,
    name: agent.name,
    task_description: agent.taskDescription,
    original_llm: agent.originalLLM,
    slm_model: agent.slmModel,
    last_updated_at: agent.updatedAt,
    last_trained_model_path: agent.lastTrainedModelPath ?? null,
    accuracy: agent.accuracy ?? null,
    model_costs_saved: agent.modelCostsSaved ?? null,
  }));

  res.json(payload);
});

router.post('/', async (req: Request, res: Response) => {
  const { id, name, taskDescription, originalLLM, tags, langfuseMetadataKey, lastTrainedModelPath, accuracy, modelCostsSaved } = req.body ?? {};

  if (typeof id !== 'string' || id.trim().length === 0) {
    return res.status(400).json({ error: 'id is required' });
  }

  if (typeof name !== 'string' || name.trim().length === 0) {
    return res.status(400).json({ error: 'name is required' });
  }

  if (typeof taskDescription !== 'string' || taskDescription.trim().length === 0) {
    return res.status(400).json({ error: 'taskDescription is required' });
  }

  if (typeof originalLLM !== 'string' || originalLLM.trim().length === 0) {
    return res.status(400).json({ error: 'originalLLM is required' });
  }

  try {
    const agent = await registerAgent({
      id: id.trim(),
      name: name.trim(),
      taskDescription: taskDescription.trim(),
      originalLLM: originalLLM.trim(),
      tags,
      langfuseMetadataKey,
      lastTrainedModelPath,
      accuracy,
      modelCostsSaved,
    });

    res.status(201).json(agent);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    const status = message.includes('already registered') ? 409 : 500;
    res.status(status).json({ error: message });
  }
});

export default router;
