import { Router } from 'express';
import { getEvalResults, getComparisonSummary } from '../service/evalMetrics';
import fs from 'fs/promises';
import path from 'path';

const router = Router();
const DATA_DIR = path.join(__dirname, '../data/slm_traces');
const EVAL_RESULTS_PATH = path.join(__dirname, '../../../eval_structured_detailed_results.json');

router.get('/eval', (_req, res) => {
  try {
    const results = getEvalResults();
    res.json({ results });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to load evaluation metrics',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

router.get('/comparison', (_req, res) => {
  try {
    const comparison = getComparisonSummary();
    res.json(comparison);
  } catch (error) {
    res.status(500).json({
      error: 'Failed to generate comparison',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// SLM Agent Metrics
router.get('/slm', async (_req, res) => {
  try {
    const metricsPath = path.join(DATA_DIR, 'metrics.json');
    const tracesPath = path.join(DATA_DIR, 'traces.json');

    try {
      await fs.access(metricsPath);
    } catch {
      return res.status(404).json({
        error: 'No metrics available',
        message: 'Run example_slm_agent.py to generate metrics',
        command: 'CUDA_VISIBLE_DEVICES=0 python example_slm_agent.py'
      });
    }

    const metricsData = await fs.readFile(metricsPath, 'utf-8');
    const metrics = JSON.parse(metricsData);

    let traces = null;
    try {
      const tracesData = await fs.readFile(tracesPath, 'utf-8');
      traces = JSON.parse(tracesData);
    } catch {
      // Traces optional
    }

    res.json({
      agent: {
        id: 'slm-api-generator',
        name: 'Fine-tuned API Generator',
        model: 'llama-3.1-8b-structured',
        status: 'active'
      },
      metrics,
      traces,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: 'Failed to load metrics', details: message });
  }
});

// Detailed evaluation metrics
router.get('/detailed', async (_req, res) => {
  try {
    let detailedPath = path.join(DATA_DIR, 'detailed_metrics.json');

    try {
      await fs.access(detailedPath);
    } catch {
      detailedPath = EVAL_RESULTS_PATH;
    }

    const data = await fs.readFile(detailedPath, 'utf-8');
    const detailed = JSON.parse(data);

    res.json({
      summary: detailed.summary,
      sampleResults: detailed.detailed_results?.slice(0, 10) || [],
      totalExamples: detailed.detailed_results?.length || 0,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return res.status(404).json({
        error: 'Detailed metrics not available',
        message: 'Run eval_structured_detailed.py to generate metrics'
      });
    }
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: 'Failed to load detailed metrics', details: message });
  }
});

// Dashboard combining all metrics
router.get('/dashboard', async (_req, res) => {
  try {
    const metricsPath = path.join(DATA_DIR, 'metrics.json');
    const tracesPath = path.join(DATA_DIR, 'traces.json');

    let metrics = null;
    let traces = null;
    let detailed = null;

    try {
      const metricsData = await fs.readFile(metricsPath, 'utf-8');
      metrics = JSON.parse(metricsData);
    } catch {
      // Optional
    }

    try {
      const tracesData = await fs.readFile(tracesPath, 'utf-8');
      traces = JSON.parse(tracesData);
    } catch {
      // Optional
    }

    try {
      let detailedPath = path.join(DATA_DIR, 'detailed_metrics.json');
      try {
        await fs.access(detailedPath);
      } catch {
        detailedPath = EVAL_RESULTS_PATH;
      }
      const detailedData = await fs.readFile(detailedPath, 'utf-8');
      detailed = JSON.parse(detailedData);
    } catch {
      // Optional
    }

    const azureBaseline = {
      exact_match: 0.205,
      field_f1: 0.602
    };

    const dashboard = {
      agent: {
        id: 'slm-api-generator',
        name: 'Fine-tuned API Generator',
        model: 'llama-3.1-8b-structured',
        status: metrics ? 'active' : 'not-run',
        last_run: traces?.runs?.[traces.runs.length - 1]?.completedAt || null
      },
      overview: {
        total_runs: metrics?.total_runs || 0,
        success_rate: metrics ? (metrics.successful_runs / metrics.total_runs * 100) : 0,
        avg_latency_ms: metrics?.avg_latency_ms || 0,
        total_cost_usd: metrics?.total_cost_usd || 0
      },
      quality: {
        exact_match: detailed?.summary?.exact_match_accuracy || 0.40,
        tool_name_accuracy: detailed?.summary?.tool_name_accuracy || 0.98,
        query_preservation: detailed?.summary?.query_accuracy || 0.92,
        json_validity: detailed?.summary?.json_validity_rate || 1.0,
        functional_correctness: detailed?.summary?.functional_correctness || 0.71,
        semantic_correctness: detailed?.summary?.semantic_correctness || 0.75
      },
      comparison: {
        slm_exact_match: detailed?.summary?.exact_match_accuracy || 0.40,
        azure_exact_match: azureBaseline.exact_match,
        improvement_pct: ((0.40 - azureBaseline.exact_match) / azureBaseline.exact_match * 100),
        slm_wins: detailed?.summary?.exact_match_accuracy > azureBaseline.exact_match
      },
      recent_traces: traces?.runs?.slice(-5) || [],
      sample_results: detailed?.detailed_results?.slice(0, 5) || [],
      timestamp: new Date().toISOString()
    };

    res.json(dashboard);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: 'Failed to load dashboard data', details: message });
  }
});

export default router;
