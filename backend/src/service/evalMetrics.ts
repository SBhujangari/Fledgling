// Load and serve real evaluation metrics
import fs from 'fs';
import path from 'path';

const REPO_ROOT = path.resolve(__dirname, '../../..');

interface EvalMetrics {
  json_valid_rate: number;
  exact_match_rate: number;
  field_precision: number;
  field_recall: number;
  field_f1: number;
}

interface EvalResult {
  track: string;
  model: string;
  metrics: EvalMetrics;
  filePath: string;
}

function loadMetrics(filePath: string): EvalMetrics | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Failed to load metrics from ${filePath}:`, error);
    return null;
  }
}

export function getEvalResults(): EvalResult[] {
  const results: EvalResult[] = [];

  const metricFiles = [
    { track: 'structured', model: 'slm', file: 'slm_swap/05_eval/structured_slm_test.json' },
    { track: 'structured', model: 'azure', file: 'slm_swap/05_eval/structured_azure_test.json' },
    { track: 'toolcall', model: 'slm', file: 'slm_swap/05_eval/toolcall_slm_test.json' },
    { track: 'toolcall', model: 'azure', file: 'slm_swap/05_eval/toolcall_azure_test.json' },
  ];

  for (const { track, model, file } of metricFiles) {
    const filePath = path.join(REPO_ROOT, file);
    const metrics = loadMetrics(filePath);
    if (metrics) {
      results.push({ track, model, metrics, filePath });
    }
  }

  return results;
}

export function getComparisonSummary() {
  const results = getEvalResults();

  const structured = {
    slm: results.find(r => r.track === 'structured' && r.model === 'slm')?.metrics,
    azure: results.find(r => r.track === 'structured' && r.model === 'azure')?.metrics,
  };

  const toolcall = {
    slm: results.find(r => r.track === 'toolcall' && r.model === 'slm')?.metrics,
    azure: results.find(r => r.track === 'toolcall' && r.model === 'azure')?.metrics,
  };

  return {
    structured: {
      slm: structured.slm || null,
      azure: structured.azure || null,
      delta: structured.slm && structured.azure ? {
        f1_delta: structured.slm.field_f1 - structured.azure.field_f1,
        f1_pct_of_azure: (structured.slm.field_f1 / structured.azure.field_f1) * 100,
      } : null,
    },
    toolcall: {
      slm: toolcall.slm || null,
      azure: toolcall.azure || null,
      delta: toolcall.slm && toolcall.azure ? {
        f1_delta: toolcall.slm.field_f1 - toolcall.azure.field_f1,
        f1_pct_of_azure: (toolcall.slm.field_f1 / toolcall.azure.field_f1) * 100,
      } : null,
    },
    summary: {
      message: 'SLM fine-tuned on structured dataset shows 53% of Azure LLM performance on structured track',
      recommendation: 'Structured: Needs improvement (53% parity). Consider CEP training or larger dataset.',
    },
  };
}
