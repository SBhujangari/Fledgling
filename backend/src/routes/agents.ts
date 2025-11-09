import { Router, Request, Response } from 'express';

import { listAgents, registerAgent } from '../service/store/agentStore';
import { registerTools } from '../service/store/toolStore';

const router = Router();

router.get('/', async (_req: Request, res: Response) => {
  const agents = await listAgents();
  const payload = agents.map((agent) => ({
    id: agent.id,
    name: agent.name,
    task_description: agent.taskDescription,
    instructions: agent.instructions,
    original_llm: agent.originalLLM,
    slm_model: agent.slmModel,
    last_updated_at: agent.updatedAt,
    last_trained_model_path: agent.lastTrainedModelPath ?? null,
    accuracy: agent.accuracy ?? null,
    model_costs_saved: agent.modelCostsSaved ?? null,
    tool_ids: agent.toolIds ?? [],
  }));

  res.json(payload);
});

router.post('/', async (req: Request, res: Response) => {
  const {
    id,
    name,
    taskDescription,
    instructions,
    originalLLM,
    tags,
    langfuseMetadataKey,
    lastTrainedModelPath,
    accuracy,
    modelCostsSaved,
    tools,
  } = req.body ?? {};

  if (typeof id !== 'string' || id.trim().length === 0) {
    return res.status(400).json({ error: 'id is required' });
  }

  if (typeof name !== 'string' || name.trim().length === 0) {
    return res.status(400).json({ error: 'name is required' });
  }

  if (typeof taskDescription !== 'string' || taskDescription.trim().length === 0) {
    return res.status(400).json({ error: 'taskDescription is required' });
  }

  if (typeof instructions !== 'string' || instructions.trim().length === 0) {
    return res.status(400).json({ error: 'instructions is required' });
  }

  if (typeof originalLLM !== 'string' || originalLLM.trim().length === 0) {
    return res.status(400).json({ error: 'originalLLM is required' });
  }

  try {
    const normalizedTools = Array.isArray(tools)
      ? tools.map((tool: any, index: number) => ({
          id:
            typeof tool?.id === 'string' && tool.id.trim().length > 0
              ? tool.id.trim()
              : `${id.trim()}::${typeof tool?.name === 'string' ? tool.name : `tool-${index + 1}`}`,
          name: typeof tool?.name === 'string' ? tool.name : undefined,
          description: typeof tool?.description === 'string' ? tool.description : undefined,
          inputSchema: tool?.inputSchema ?? tool?.parameters ?? undefined,
          outputSchema: tool?.outputSchema,
          metadata: tool?.metadata ?? null,
        }))
      : undefined;

    const registeredTools = await registerTools(normalizedTools);
    const toolIds = registeredTools.map((tool) => tool.id);

    const agent = await registerAgent({
      id: id.trim(),
      name: name.trim(),
      taskDescription: taskDescription.trim(),
      instructions: instructions.trim(),
      originalLLM: originalLLM.trim(),
      tags,
      langfuseMetadataKey,
      lastTrainedModelPath,
      accuracy,
      modelCostsSaved,
      toolIds,
    });

    res.status(201).json(agent);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    const status = message.includes('already registered') ? 409 : 500;
    res.status(status).json({ error: message });
  }
});

export default router;
