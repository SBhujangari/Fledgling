import { Router, Request, Response } from 'express';

import { getAgentById } from '../service/store/agentStore';
import { listAllTools, listToolsByIds, registerTools } from '../service/store/toolStore';

const router = Router();

router.get('/', async (req: Request, res: Response) => {
  const agentId = typeof req.query.agentId === 'string' ? req.query.agentId : undefined;

  if (agentId) {
    const agent = await getAgentById(agentId);
    if (!agent) {
      return res.status(404).json({ error: `Agent with id "${agentId}" not found` });
    }

    const tools = await listToolsByIds(agent.toolIds ?? []);
    return res.json(tools);
  }

  const tools = await listAllTools();
  return res.json(tools);
});

router.post('/', async (req: Request, res: Response) => {
  const toolsInput = Array.isArray(req.body) ? req.body : req.body?.tools;
  if (!Array.isArray(toolsInput) || toolsInput.length === 0) {
    return res.status(400).json({ error: 'tools array is required' });
  }

  try {
    const results = await registerTools(toolsInput);
    res.status(201).json(results);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(400).json({ error: message });
  }
});

export default router;
