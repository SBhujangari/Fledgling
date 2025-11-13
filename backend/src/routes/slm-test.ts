import { Router } from 'express';
import { spawn } from 'child_process';
import path from 'path';

const router = Router();

// Store for active inference sessions
const activeInferences = new Map<string, {
  process: any;
  output: string[];
  status: 'running' | 'completed' | 'error';
  startTime: number;
}>();

/**
 * POST /api/slm-test/inference
 * Test the fine-tuned SLM model with a prompt
 */
router.post('/inference', async (req, res) => {
  const { prompt, model = 'structured' } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  const inferenceId = `inf_${Date.now()}_${Math.random().toString(36).substring(7)}`;

  // Prepare test script
  const scriptPath = path.join(__dirname, '../../..', 'test_single_inference.py');

  try {
    res.json({
      inferenceId,
      status: 'initiated',
      message: 'Inference started. Use GET /api/slm-test/inference/:id to check status'
    });

    // Run inference in background
    const pythonProcess = spawn('python', [scriptPath, prompt, model], {
      env: {
        ...process.env,
        CUDA_VISIBLE_DEVICES: '0',
        PYTHONUNBUFFERED: '1'
      }
    });

    activeInferences.set(inferenceId, {
      process: pythonProcess,
      output: [],
      status: 'running',
      startTime: Date.now()
    });

    pythonProcess.stdout.on('data', (data: Buffer) => {
      const inference = activeInferences.get(inferenceId);
      if (inference) {
        inference.output.push(data.toString());
      }
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      const inference = activeInferences.get(inferenceId);
      if (inference) {
        inference.output.push(`[ERROR] ${data.toString()}`);
      }
    });

    pythonProcess.on('close', (code: number) => {
      const inference = activeInferences.get(inferenceId);
      if (inference) {
        inference.status = code === 0 ? 'completed' : 'error';
      }
    });

  } catch (error: any) {
    return res.status(500).json({
      error: 'Failed to start inference',
      details: error.message
    });
  }
});

/**
 * GET /api/slm-test/inference/:id
 * Get status and results of an inference job
 */
router.get('/inference/:id', (req, res) => {
  const { id } = req.params;
  const inference = activeInferences.get(id);

  if (!inference) {
    return res.status(404).json({ error: 'Inference job not found' });
  }

  const duration = Date.now() - inference.startTime;

  // Try to extract JSON result from output
  let result = null;
  let fullOutput = inference.output.join('');

  // Look for JSON in output
  const jsonMatch = fullOutput.match(/\{[^{}]*"arguments"[^{}]*\}/);
  if (jsonMatch) {
    try {
      result = JSON.parse(jsonMatch[0]);
    } catch (e) {
      // If parsing fails, keep as string
      result = jsonMatch[0];
    }
  }

  res.json({
    inferenceId: id,
    status: inference.status,
    duration: `${(duration / 1000).toFixed(2)}s`,
    result,
    fullOutput: inference.status === 'completed' ? fullOutput : undefined
  });

  // Clean up completed inferences after 5 minutes
  if (inference.status !== 'running' && duration > 300000) {
    activeInferences.delete(id);
  }
});

/**
 * POST /api/slm-test/compare
 * Compare SLM output vs Azure GPT baseline
 */
router.post('/compare', async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  const comparisonId = `cmp_${Date.now()}`;

  try {
    // Run both inferences in parallel
    const scriptPath = path.join(__dirname, '../../..', 'test_comparison.py');

    const pythonProcess = spawn('python', [scriptPath, prompt], {
      env: {
        ...process.env,
        CUDA_VISIBLE_DEVICES: '0',
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
        try {
          // Parse comparison results
          const jsonMatch = output.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const results = JSON.parse(jsonMatch[0]);
            res.json({
              comparisonId,
              ...results,
              timestamp: new Date().toISOString()
            });
          } else {
            res.status(500).json({
              error: 'Failed to parse comparison results',
              output
            });
          }
        } catch (error: any) {
          res.status(500).json({
            error: 'Failed to process comparison',
            details: error.message,
            output
          });
        }
      } else {
        res.status(500).json({
          error: 'Comparison process failed',
          stderr: errorOutput,
          stdout: output
        });
      }
    });

  } catch (error: any) {
    return res.status(500).json({
      error: 'Failed to start comparison',
      details: error.message
    });
  }
});

/**
 * GET /api/slm-test/metrics
 * Get current evaluation metrics
 */
router.get('/metrics', async (req, res) => {
  const fs = require('fs').promises;
  const metricsPath = path.join(__dirname, '../../..', 'eval_structured_detailed_results.json');

  try {
    const data = await fs.readFile(metricsPath, 'utf8');
    const metrics = JSON.parse(data);

    res.json({
      timestamp: new Date().toISOString(),
      summary: metrics.summary,
      sampleResults: metrics.detailed_results.slice(0, 5) // Return first 5 examples
    });
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      res.status(404).json({
        error: 'Metrics not yet available',
        message: 'Run eval_structured_detailed.py first to generate metrics'
      });
    } else {
      res.status(500).json({
        error: 'Failed to load metrics',
        details: error.message
      });
    }
  }
});

/**
 * GET /api/slm-test/model-info
 * Get information about the deployed model
 */
router.get('/model-info', (req, res) => {
  res.json({
    model: {
      name: 'Llama 3.1 8B - Structured API Adapter',
      baseModel: 'unsloth/llama-3.1-8b-instruct-bnb-4bit',
      adapterPath: 'slm_swap/04_ft/adapter_llama_structured',
      huggingFace: 'kineticdrive/llama-structured-api-adapter',
      size: '335MB (adapter only)',
      quantization: '4-bit',
      trainableParams: '84M (1.04%)'
    },
    performance: {
      exactMatchAccuracy: '40.0%',
      azureBaseline: '20.5%',
      improvement: '+95%',
      toolNameAccuracy: '92%+',
      argsPartialMatch: '76%+',
      jsonValidity: '100%',
      functionalCorrectness: '84%+'
    },
    training: {
      duration: '4m 52s',
      trainingLoss: 0.50,
      validationLoss: 0.58,
      epochs: 3,
      dataset: {
        training: 300,
        validation: 60,
        test: 50
      }
    },
    deployment: {
      status: 'production-ready',
      gpu: '2x RTX 3090',
      inferenceTime: '~5-7s per request',
      batchCapable: true
    }
  });
});

/**
 * GET /api/slm-test/health
 * Check if the model is loaded and ready
 */
router.get('/health', async (req, res) => {
  const { spawn } = require('child_process');

  const healthCheck = spawn('python', ['-c', 'import torch; print(torch.cuda.is_available())']);

  let output = '';
  healthCheck.stdout.on('data', (data: Buffer) => {
    output += data.toString();
  });

  healthCheck.on('close', (code: number) => {
    const cudaAvailable = output.trim() === 'True';

    res.json({
      status: code === 0 && cudaAvailable ? 'healthy' : 'degraded',
      cuda: cudaAvailable,
      gpuCount: cudaAvailable ? 'Available' : 'Not available',
      modelLoaded: false, // Would need to check actual model state
      timestamp: new Date().toISOString()
    });
  });
});

export default router;
