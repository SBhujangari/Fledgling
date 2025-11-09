import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SCRIPT_PATH = path.join(REPO_ROOT, 'slm_swap', 'hf_upload.py');
const VALID_REPO_TYPES = new Set(['model', 'dataset', 'space'] as const);

export type HuggingFaceRepoType = 'model' | 'dataset' | 'space';

export interface HuggingFaceUploadOptions {
  paths: string[];
  repoId: string;
  repoType?: HuggingFaceRepoType;
  private?: boolean;
  commitMessage?: string;
  branch?: string;
  pathInRepo?: string;
  autoSubdir?: boolean;
  token?: string;
}

export interface HuggingFaceUploadResult {
  repoId: string;
  repoType: HuggingFaceRepoType;
  private: boolean;
  paths: string[];
  command: string[];
  stdout: string;
  stderr: string;
  durationMs: number;
}

function ensureScriptAvailable(): void {
  if (!fs.existsSync(SCRIPT_PATH)) {
    throw new Error(`Missing helper script at ${SCRIPT_PATH}. Run the uploader from the repo root.`);
  }
}

function resolveRepoPath(inputPath: string): string {
  const resolved = path.resolve(REPO_ROOT, inputPath);
  const relative = path.relative(REPO_ROOT, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Path "${inputPath}" resolves outside the repository root.`);
  }
  if (!fs.existsSync(resolved)) {
    throw new Error(`Path "${inputPath}" does not exist on disk at ${resolved}.`);
  }
  return resolved;
}

function normalizeRepoId(repoId: string): string {
  const trimmed = repoId.trim();
  const repoIdPattern = /^[\w.-]+\/[\w.-]+$/;
  if (!repoIdPattern.test(trimmed)) {
    throw new Error('Repo ID must look like "owner/name" and contain only letters, numbers, "-", "_" or ".".');
  }
  return trimmed;
}

async function runProcess(command: string, args: string[], env: NodeJS.ProcessEnv): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: REPO_ROOT,
      env,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      resolve({ stdout, stderr, exitCode: code ?? -1 });
    });
  });
}

export async function uploadToHuggingFace(options: HuggingFaceUploadOptions): Promise<HuggingFaceUploadResult> {
  if (!options.paths?.length) {
    throw new Error('At least one path is required to run the upload.');
  }

  ensureScriptAvailable();

  const token = options.token?.trim() ?? process.env.HF_UPLOAD_TOKEN?.trim() ?? process.env.HUGGING_FACE_HUB_TOKEN?.trim();
  if (!token) {
    throw new Error('Missing Hugging Face token. Set HF_UPLOAD_TOKEN or HUGGING_FACE_HUB_TOKEN in backend/.env.');
  }

  const repoId = normalizeRepoId(options.repoId);
  const repoType: HuggingFaceRepoType = options.repoType ?? 'model';
  if (!VALID_REPO_TYPES.has(repoType)) {
    throw new Error(`Invalid repo type "${repoType}". Expected one of ${Array.from(VALID_REPO_TYPES).join(', ')}.`);
  }

  const resolvedPaths = options.paths.map(resolveRepoPath);
  const pythonExec = process.env.HF_UPLOAD_PYTHON?.trim() || 'python';

  const args: string[] = [SCRIPT_PATH, ...resolvedPaths, '--repo-id', repoId, '--repo-type', repoType];
  if (options.private) {
    args.push('--private');
  }
  const commitMessage = options.commitMessage?.trim();
  if (commitMessage) {
    args.push('--commit-message', commitMessage);
  }
  const branch = options.branch?.trim();
  if (branch) {
    args.push('--branch', branch);
  }
  const pathInRepo = options.pathInRepo?.trim();
  if (pathInRepo) {
    args.push('--path-in-repo', pathInRepo);
  }
  if (options.autoSubdir) {
    args.push('--auto-subdir');
  }

  const env = {
    ...process.env,
    HUGGING_FACE_HUB_TOKEN: token,
  };

  const start = performance.now();
  const { stdout, stderr, exitCode } = await runProcess(pythonExec, args, env);
  const durationMs = performance.now() - start;

  if (exitCode !== 0) {
    throw new Error(stderr.trim() || `Uploader exited with code ${exitCode}`);
  }

  return {
    repoId,
    repoType,
    private: Boolean(options.private),
    paths: resolvedPaths,
    command: [pythonExec, ...args],
    stdout,
    stderr,
    durationMs,
  };
}
