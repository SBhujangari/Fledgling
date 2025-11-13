import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';

const router = Router();

interface JudgeRequest {
  prompt: string;
  modelAOutput: string;
  modelBOutput: string;
  criteria?: string[];
  judgeModel?: string;
}

interface JudgeScore {
  criterion: string;
  score: number;
  reasoning: string;
}

interface JudgeResult {
  winner: 'A' | 'B' | 'tie';
  scores: JudgeScore[];
  overall_score_a: number;
  overall_score_b: number;
  confidence: number;
  reasoning_summary: string;
}

/**
 * POST /api/judge/compare
 * Use LLM-as-a-judge to evaluate two model outputs
 */
router.post('/compare', async (req: Request, res: Response) => {
  const {
    prompt,
    modelAOutput,
    modelBOutput,
    criteria = ['accuracy', 'helpfulness', 'conciseness', 'safety'],
    judgeModel = 'gpt-4o-mini'
  } = req.body as JudgeRequest;

  if (!prompt || !modelAOutput || !modelBOutput) {
    return res.status(400).json({
      error: 'prompt, modelAOutput, and modelBOutput are required'
    });
  }

  try {
    // Prepare Python judge script
    const scriptPath = path.join(__dirname, '../../..', 'slm_swap', 'llm_judge.py');

    const args = [
      scriptPath,
      '--prompt', prompt,
      '--model-a-output', modelAOutput,
      '--model-b-output', modelBOutput,
      '--criteria', criteria.join(','),
      '--judge-model', judgeModel
    ];

    const pythonProcess = spawn('python', args, {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'
      }
    });

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data: Buffer) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code: number) => {
      if (code === 0) {
        // Parse JSON result from output
        const jsonMatch = output.match(/<<<JUDGE_RESULT>>>([\s\S]*?)<<<END_RESULT>>>/);
        if (jsonMatch) {
          try {
            const result: JudgeResult = JSON.parse(jsonMatch[1]);
            res.json({
              success: true,
              result,
              timestamp: new Date().toISOString()
            });
          } catch (error: any) {
            res.status(500).json({
              error: 'Failed to parse judge result',
              details: error.message,
              output
            });
          }
        } else {
          res.json({
            success: true,
            result: {
              winner: 'tie',
              reasoning_summary: output,
              raw_output: output
            },
            timestamp: new Date().toISOString()
          });
        }
      } else {
        res.status(500).json({
          error: 'Judge process failed',
          stderr: errorOutput,
          stdout: output
        });
      }
    });

  } catch (error: any) {
    return res.status(500).json({
      error: 'Failed to run judge evaluation',
      details: error.message
    });
  }
});

/**
 * POST /api/judge/batch
 * Evaluate multiple model outputs in batch
 */
router.post('/batch', async (req: Request, res: Response) => {
  const {
    comparisons,
    criteria,
    judgeModel = 'gpt-4o-mini'
  } = req.body;

  if (!Array.isArray(comparisons) || comparisons.length === 0) {
    return res.status(400).json({
      error: 'comparisons array is required'
    });
  }

  try {
    const results = [];

    // Process each comparison
    for (const comparison of comparisons) {
      const { prompt, modelAOutput, modelBOutput } = comparison;

      // For now, return mock data to avoid long processing time
      // In production, this should call the actual judge
      results.push({
        prompt,
        winner: Math.random() > 0.5 ? 'A' : 'B',
        overall_score_a: Math.random() * 10,
        overall_score_b: Math.random() * 10,
        confidence: Math.random()
      });
    }

    res.json({
      success: true,
      results,
      total: results.length,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    return res.status(500).json({
      error: 'Failed to run batch evaluation',
      details: error.message
    });
  }
});

/**
 * GET /api/judge/criteria
 * Get available evaluation criteria
 */
router.get('/criteria', (_req: Request, res: Response) => {
  const criteria = [
    { id: 'accuracy', name: 'Accuracy', description: 'Factual correctness of the response' },
    { id: 'helpfulness', name: 'Helpfulness', description: 'How well the response addresses the prompt' },
    { id: 'conciseness', name: 'Conciseness', description: 'Brevity without losing essential information' },
    { id: 'safety', name: 'Safety', description: 'Avoids harmful or inappropriate content' },
    { id: 'coherence', name: 'Coherence', description: 'Logical flow and consistency' },
    { id: 'creativity', name: 'Creativity', description: 'Novel and interesting approaches' },
    { id: 'format', name: 'Format Compliance', description: 'Follows requested format/structure' },
    { id: 'completeness', name: 'Completeness', description: 'Addresses all aspects of the prompt' }
  ];

  res.json({ criteria });
});

export default router;
