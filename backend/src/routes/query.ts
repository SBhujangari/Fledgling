import { Router, Request, Response } from 'express';

const router = Router();

/**
 * POST /query
 * Execute agent task with the provided query/task
 *
 * Body:
 * - task: string - The task/query to execute
 * - agentId?: string - Optional agent ID to use
 * - model?: string - Optional model to use (defaults to agent's trained model or base model)
 */
router.post('/query', async (req: Request, res: Response) => {
  try {
    const { task, agentId, model } = req.body ?? {};

    if (!task || typeof task !== 'string') {
      return res.status(400).json({
        error: 'task is required and must be a string',
        example: {
          task: 'Your task description here',
          agentId: 'optional-agent-id',
          model: 'optional-model-name'
        }
      });
    }

    // TODO: Implement actual agent execution
    // This should:
    // 1. Load the specified agent (or default agent)
    // 2. Execute the task using the agent's model
    // 3. Track the execution as a trace
    // 4. Return the result

    console.log('Query received:', { task, agentId, model });

    // Placeholder response
    res.json({
      status: 'success',
      message: 'Query endpoint is set up. Agent execution to be implemented.',
      request: {
        task: task.substring(0, 100) + (task.length > 100 ? '...' : ''),
        agentId: agentId || 'default',
        model: model || 'default',
      },
      todo: [
        'Implement agent model loading',
        'Execute task with model',
        'Track execution as trace',
        'Return actual results'
      ]
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

export default router;
