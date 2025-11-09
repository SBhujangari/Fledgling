import { Router, Request, Response } from 'express';

import { fetchCompletedTraces } from '../service/loader';
import { transformTraces, type TransformResult } from '../service/transformer';
import { parseTraceToSample } from '../parsers/otelParser';
import { ensureAgentRegistered, updateAgentMetrics, getAgentById } from '../service/store/agentStore';
import type { FinetuneSample } from '../types/finetune';
import { calculateCostSavings, computeModelAccuracy } from '../utils/costCalculator';

const router = Router();

router.get('/traces', async (req: Request, res: Response) => {
  try {
    const updatedAfter = typeof req.query.updatedAfter === 'string' ? req.query.updatedAfter : undefined;
    const aggregate: TransformResult = {
      runs: [],
      observations: [],
      generations: [],
    };
    const samples: FinetuneSample[] = [];

    for await (const page of fetchCompletedTraces({ updatedAfter })) {
      const transformed = transformTraces(page);
      aggregate.runs.push(...transformed.runs);
      aggregate.observations.push(...transformed.observations);
      aggregate.generations.push(...transformed.generations);

      for (const trace of page) {
        const sample = parseTraceToSample(trace);
        if (sample) {
          // Only process traces for registered agents
          const agent = await ensureAgentRegistered(sample.agentId);
          if (agent) {
            samples.push(sample);
          }
          // Silently skip traces for unregistered agents
        }
      }
    }

    res.json({ ...aggregate, samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

router.post('/train', async (req: Request, res: Response) => {
  try {
    const {
      agentId,
      llmModel = 'gpt-4',
      slmModel = 'llama-3-8b',
      inputTokens = 100000,  // Default: 100k tokens
      outputTokens = 50000,  // Default: 50k tokens
    } = req.body ?? {};

    if (!agentId || typeof agentId !== 'string') {
      return res.status(400).json({ error: 'agentId is required' });
    }

    // Ensure agent exists
    const agent = await getAgentById(agentId);
    if (!agent) {
      return res.status(404).json({ error: `Agent with id "${agentId}" not found` });
    }

    // Calculate cost savings based on token usage
    const costSavings = calculateCostSavings({
      llmModel,
      slmModel,
      tokenUsage: {
        inputTokens,
        outputTokens,
      },
    });

    // TODO: Compute actual model accuracy from evaluation dataset
    // For now, using placeholder function
    const accuracy = computeModelAccuracy(agent.lastTrainedModelPath ?? '');

    // Update agent metrics
    const updatedAgent = await updateAgentMetrics(agentId, {
      accuracy,
      modelCostsSaved: (agent.modelCostsSaved ?? 0) + costSavings,
    });

    res.json({
      status: 'success',
      message: 'Training metrics computed and saved',
      metrics: {
        costSavings,
        totalCostsSaved: updatedAgent.modelCostsSaved,
        accuracy: updatedAgent.accuracy,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

export default router;
