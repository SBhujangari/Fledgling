import { Router, Request, Response } from 'express';

import { ensureAgentRegistered, getAgentById, updateAgentMetrics } from '../service/store/agentStore';
import { calculateCostSavings, computeModelAccuracy } from '../utils/costCalculator';
import { getTrainingStatus } from '../service/trainingStatus';

const router = Router();

router.post('/', async (req: Request, res: Response) => {
  try {
    const {
      agentId,
      llmModel = 'gpt-4o-mini',
      slmModel = 'qwen2.5-7b-instruct',
      inputTokens = 100000,
      outputTokens = 50000,
    } = req.body ?? {};

    if (!agentId || typeof agentId !== 'string') {
      return res.status(400).json({ error: 'agentId is required' });
    }

    const agent = await getAgentById(agentId);
    if (!agent) {
      return res.status(404).json({ error: `Agent ${agentId} is not registered` });
    }

    const costSavings = calculateCostSavings({
      llmModel,
      slmModel,
      tokenUsage: {
        inputTokens,
        outputTokens,
      },
    });

    const accuracy = computeModelAccuracy(agent.lastTrainedModelPath ?? '');
    const updatedAgent = await updateAgentMetrics(agentId, {
      accuracy,
      modelCostsSaved: (agent.modelCostsSaved ?? 0) + costSavings,
    });

    res.json({
      status: 'queued',
      message: 'Fine-tune pipeline triggered (mock). Monitor Langfuse for real progress.',
      metrics: {
        accuracy: updatedAgent.accuracy,
        totalCostsSaved: updatedAgent.modelCostsSaved,
        latestCostSavings: costSavings,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown training error';
    res.status(500).json({ error: message });
  }
});

router.get('/status', async (_req: Request, res: Response) => {
  try {
    const status = await getTrainingStatus();
    res.json(status);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error while reading training status.';
    res.status(500).json({ status: 'error', message });
  }
});

export default router;
