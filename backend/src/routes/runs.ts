import { Router, Request, Response } from 'express';
import { listRuns, createRun, updateRunStatus } from '../service/store/runStore';
import { getAgentById, updateAgentTrainingStatus, updateAgentTrainingMetrics } from '../service/store/agentStore';

const router = Router();

router.get('/', async (req: Request, res: Response) => {
  const agentId = typeof req.query.agentId === 'string' ? req.query.agentId : undefined;
  const runs = await listRuns(agentId);
  res.json(runs);
});

router.post('/', async (req: Request, res: Response) => {
  const { agentId, name } = req.body ?? {};

  if (typeof agentId !== 'string' || agentId.trim().length === 0) {
    return res.status(400).json({ error: 'agentId is required' });
  }

  if (typeof name !== 'string' || name.trim().length === 0) {
    return res.status(400).json({ error: 'name is required' });
  }

  try {
    const agent = await getAgentById(agentId);
    if (!agent) {
      return res.status(404).json({ error: `Agent id "${agentId}" not found` });
    }

    const run = await createRun(agentId.trim(), name.trim());
    
    // Update agent training status to true
    await updateAgentTrainingStatus(agentId.trim(), true);
    
    // Update run status to running immediately
    await updateRunStatus(run.id, 'running');

    // Start background training process
    setTimeout(async () => {
      try {
        // TODO: Replace this random generation with actual API call to Gabe's publishable link once ready
        // For now, generate random samples (0-50) to feed into training
        const newSamples = Math.floor(Math.random() * 51); // 0-50
        
        const currentAccuracy = agent.accuracy ?? 0;
        const currentCostSavings = agent.modelCostsSaved ?? 0;
        const currentIterations = agent.iterations ?? 1;
        const currentTrainingDataSize = agent.trainingDataSize ?? 50;
        
        // Random accuracy increase: 5-10%
        const accuracyIncrease = Math.random() * 5 + 5; // 5-10
        const newAccuracy = Math.min(100, currentAccuracy + accuracyIncrease);
        
        // Random cost savings increase: 1-5%
        const costSavingsIncrease = (currentCostSavings * (Math.random() * 0.04 + 0.01)); // 1-5% of current
        const newCostSavings = currentCostSavings + costSavingsIncrease;
        
        // Increment iterations by 1
        const newIterations = currentIterations + 1;
        
        // Add new samples to training data size (cumulative)
        const newTrainingDataSize = currentTrainingDataSize + newSamples;
        
        // Update agent metrics
        await updateAgentTrainingMetrics(agentId.trim(), {
          accuracy: newAccuracy,
          modelCostsSaved: newCostSavings,
          iterations: newIterations,
          trainingDataSize: newTrainingDataSize,
        });
        
        // Update run status to completed
        await updateRunStatus(run.id, 'completed');
        
        // Set agent training status to false
        await updateAgentTrainingStatus(agentId.trim(), false);
      } catch (error) {
        console.error('Error updating training metrics:', error);
        // Update run status to failed on error
        await updateRunStatus(run.id, 'failed');
        await updateAgentTrainingStatus(agentId.trim(), false);
      }
    }, 30000); // 30 second delay

    res.status(201).json(run);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

router.patch('/:id', async (req: Request, res: Response) => {
  const { id } = req.params;
  const { status, completedAt } = req.body ?? {};

  if (!status || !['queued', 'running', 'completed', 'failed'].includes(status)) {
    return res.status(400).json({ error: 'Valid status is required' });
  }

  try {
    const run = await updateRunStatus(id, status, completedAt ?? undefined);
    res.json(run);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    const statusCode = message.includes('not found') ? 404 : 500;
    res.status(statusCode).json({ error: message });
  }
});

export default router;

