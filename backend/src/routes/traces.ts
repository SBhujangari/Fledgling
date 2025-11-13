import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';

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
          const agent = await ensureAgentRegistered(sample.agentId);
          if (agent) {
            samples.push(sample);
          }
        }
      }
    }

    res.json({ ...aggregate, samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    if (message.includes('Missing LANGFUSE')) {
      const fallback = buildMockTracePayload();
      return res.json(fallback);
    }
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

// New endpoint to get detailed trace data from local storage
router.get('/traces/local', async (req: Request, res: Response) => {
  try {
    const tracesPath = path.resolve(process.cwd(), '..', 'storage', 'dummy_langfuse_traces.jsonl');

    if (!fs.existsSync(tracesPath)) {
      return res.json({ traces: [] });
    }

    const fileContent = fs.readFileSync(tracesPath, 'utf-8');
    const traces: any[] = [];

    for (const line of fileContent.split('\n')) {
      if (!line.trim()) continue;
      try {
        traces.push(JSON.parse(line));
      } catch (err) {
        console.error('Failed to parse trace line:', err);
      }
    }

    res.json({ traces });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

// New endpoint to get a single trace by ID
router.get('/traces/local/:traceId', async (req: Request, res: Response) => {
  try {
    const { traceId } = req.params;
    const tracesPath = path.resolve(process.cwd(), '..', 'storage', 'dummy_langfuse_traces.jsonl');

    if (!fs.existsSync(tracesPath)) {
      return res.status(404).json({ error: 'Trace not found' });
    }

    const fileContent = fs.readFileSync(tracesPath, 'utf-8');

    for (const line of fileContent.split('\n')) {
      if (!line.trim()) continue;
      try {
        const trace = JSON.parse(line);
        if (trace.id === traceId) {
          return res.json(trace);
        }
      } catch (err) {
        console.error('Failed to parse trace line:', err);
      }
    }

    res.status(404).json({ error: 'Trace not found' });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

export default router;

function buildMockTracePayload(): TransformResult & { samples: FinetuneSample[] } {
  const datasetPath = path.resolve(process.cwd(), '..', 'datasets', 'biz_entity_location_date_structure.jsonl');
  const now = new Date().toISOString();
  const rows: Array<{ input: string; output: { entity: string; location: string; date: string } }> = [];
  if (fs.existsSync(datasetPath)) {
    const file = fs.readFileSync(datasetPath, 'utf-8');
    for (const line of file.split('\n')) {
      if (!line.trim()) continue;
      try {
        rows.push(JSON.parse(line));
      } catch {
        continue;
      }
      if (rows.length >= 3) break;
    }
  } else {
    rows.push(
      {
        input: 'Acme Robotics opened a new office in Tokyo on April 1st, 2025.',
        output: { entity: 'Acme Robotics', location: 'Tokyo', date: '2025-04-01' },
      },
      {
        input: 'Globex launched its Sao Paulo hub on June 5, 2024.',
        output: { entity: 'Globex', location: 'Sao Paulo', date: '2024-06-05' },
      },
    );
  }

  const samples: FinetuneSample[] = rows.map((row, index) => ({
    traceId: `mock-trace-${index + 1}`,
    agentId: 'cookbook-biz-entity',
    conversation: [
      {
        role: 'system',
        content: 'Extract entity/location/date as JSON.',
      },
      {
        role: 'user',
        content: row.input,
      },
    ],
    steps: [
      {
        type: 'thought',
        content: 'Identify the company, location, and normalize the date.',
      },
      {
        type: 'generation',
        content: JSON.stringify(row.output),
      },
    ],
    finalResponse: JSON.stringify(row.output),
  }));

  return {
    runs: samples.map((sample, index) => ({
      traceId: sample.traceId,
      name: 'mock-structured-run',
      agentId: sample.agentId,
      status: 'completed',
      startedAt: now,
      completedAt: now,
      latencyMs: 250 + index * 5,
      costUsd: 0.0005,
    })),
    observations: samples.map((sample, index) => ({
      observationId: `${sample.traceId}-obs`,
      traceId: sample.traceId,
      type: 'generation',
      name: 'structured-output',
      status: 'OK',
      startedAt: now,
      completedAt: now,
      input: sample.conversation.map((msg) => msg.content).join('\n\n'),
      output: sample.finalResponse,
      metadata: { mock: true, index },
    })),
    generations: samples.map((sample) => ({
      generationId: `${sample.traceId}-gen`,
      traceId: sample.traceId,
      observationId: `${sample.traceId}-obs`,
      model: 'mock-slm',
      prompt: sample.conversation,
      completion: sample.finalResponse,
      usage: { totalTokens: 128, inputTokens: 64, outputTokens: 64 },
      metadata: { mock: true },
    })),
    samples,
  };
}
