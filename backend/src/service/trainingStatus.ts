import fs from 'fs/promises';
import path from 'path';

export interface TrainingGpuInfo {
  name?: string;
  memoryTotalGb?: number | null;
}

export interface TrainingHardwareInfo {
  summary?: string | null;
  gpuCount?: number | null;
  cudaVisibleDevices?: string | null;
  gpus?: TrainingGpuInfo[];
}

export interface TrainingStatus {
  status: string;
  message?: string;
  logFile?: string;
  sourcePath?: string;
  pid?: number | null;
  pidAlive?: boolean;
  hardware?: TrainingHardwareInfo;
  updatedAt?: string;
  logUpdatedAt?: string | null;
  logStalenessSeconds?: number | null;
  currentStep?: number;
  totalSteps?: number;
  percentComplete?: number;
  elapsedSeconds?: number;
  remainingSeconds?: number;
  remainingDisplay?: string;
  avgStepSeconds?: number;
  stepTimeDisplay?: string;
  stepsPerMinute?: number;
  eta?: string | null;
  etaTimestamp?: number | null;
  rawLine?: string;
}

const DEFAULT_RELATIVE_PATHS = [
  path.resolve(process.cwd(), 'slm_swap', 'logs', 'finetune_progress.json'),
  path.resolve(process.cwd(), '..', 'slm_swap', 'logs', 'finetune_progress.json'),
];

function candidatePaths(): string[] {
  const explicit = process.env.FINETUNE_PROGRESS_FILE;
  const candidates = explicit ? [explicit] : [];
  candidates.push(...DEFAULT_RELATIVE_PATHS);
  // Remove duplicates while preserving order.
  return [...new Set(candidates)];
}

async function readFirstAvailable(): Promise<{ path: string; data: string } | null> {
  const candidates = candidatePaths();
  for (const candidate of candidates) {
    try {
      const data = await fs.readFile(candidate, 'utf-8');
      return { path: candidate, data };
    } catch {
      // Try the next candidate.
    }
  }
  return null;
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const candidatesTried = candidatePaths();
  const result = await readFirstAvailable();
  if (!result) {
    return {
      status: 'unknown',
      message: `Training progress snapshot not found. Run "python slm_swap/finetune_progress.py --watch" to begin tracking.`,
      sourcePath: candidatesTried[0],
      updatedAt: new Date().toISOString(),
    };
  }

  try {
    const parsed = JSON.parse(result.data) as TrainingStatus;
    return {
      ...parsed,
      sourcePath: result.path,
    };
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to parse progress snapshot (${error instanceof Error ? error.message : 'unknown error'}).`,
      sourcePath: result.path,
      updatedAt: new Date().toISOString(),
    };
  }
}
