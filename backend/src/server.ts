import './env';

import express, { type Request, type Response } from 'express';
import cors from 'cors';

import { fetchCompletedTraces } from './service/loader';
import { transformTraces, type TransformResult } from './service/transformer';
import { parseTraceToSample } from './parsers/otelParser';
import type { FinetuneSample } from './types/finetune';
import { uploadToHuggingFace, type HuggingFaceRepoType } from './service/hfUploader';
import { getSelectedModel, listModels, selectModel } from './service/slmCatalog';
import { deleteStoredHfToken, getStoredTokenMetadata, writeStoredHfToken } from './service/hfTokenStore';

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

app.get('/api/traces', async (req: Request, res: Response) => {
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
          samples.push(sample);
        }
      }
    }

    res.json({ ...aggregate, samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

interface HuggingFaceUploadRequestBody {
  repoId?: string;
  paths?: unknown;
  repoType?: HuggingFaceRepoType;
  commitMessage?: string;
  branch?: string;
  pathInRepo?: string;
  private?: boolean;
  autoSubdir?: boolean;
}

app.post('/api/hf/upload', async (req: Request<unknown, unknown, HuggingFaceUploadRequestBody>, res: Response) => {
  try {
    const body = req.body ?? {};
    const repoId = typeof body.repoId === 'string' ? body.repoId.trim() : '';
    if (!repoId) {
      res.status(400).json({ error: 'repoId is required.' });
      return;
    }

    const paths = Array.isArray(body.paths) ? body.paths.filter((p: unknown): p is string => typeof p === 'string' && p.trim().length > 0) : [];
    if (paths.length === 0) {
      res.status(400).json({ error: 'At least one path must be provided.' });
      return;
    }

    const repoType = typeof body.repoType === 'string' ? body.repoType : undefined;
    const commitMessage = typeof body.commitMessage === 'string' ? body.commitMessage : undefined;
    const branch = typeof body.branch === 'string' ? body.branch : undefined;
    const pathInRepo = typeof body.pathInRepo === 'string' ? body.pathInRepo : undefined;

    const result = await uploadToHuggingFace({
      paths,
      repoId,
      repoType,
      private: body.private === undefined ? undefined : Boolean(body.private),
      commitMessage,
      branch,
      pathInRepo,
      autoSubdir: body.autoSubdir === undefined ? undefined : Boolean(body.autoSubdir),
    });

    res.json({ ok: true, result });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ ok: false, error: message });
  }
});

app.get('/api/hf/token', (_req: Request, res: Response) => {
  const meta = getStoredTokenMetadata();
  res.json(meta);
});

interface HfTokenRequestBody {
  token?: string;
}

app.post('/api/hf/token', (req: Request<unknown, unknown, HfTokenRequestBody>, res: Response) => {
  try {
    const token = typeof req.body?.token === 'string' ? req.body.token.trim() : '';
    if (!token) {
      res.status(400).json({ error: 'token is required.' });
      return;
    }
    const meta = writeStoredHfToken(token);
    res.json({ ok: true, updatedAt: meta.updatedAt });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ ok: false, error: message });
  }
});

app.delete('/api/hf/token', (_req: Request, res: Response) => {
  deleteStoredHfToken();
  res.json({ ok: true });
});

app.get('/api/slm/models', (_req: Request, res: Response) => {
  try {
    const models = listModels();
    const selection = getSelectedModel();
    res.json({
      models,
      selectedModelId: selection?.modelId ?? null,
      selectedAt: selection?.selectedAt ?? null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ error: message });
  }
});

interface SlmSelectionRequestBody {
  modelId?: string;
}

app.post('/api/slm/select', (req: Request<unknown, unknown, SlmSelectionRequestBody>, res: Response) => {
  try {
    const modelId = typeof req.body?.modelId === 'string' ? req.body.modelId.trim() : '';
    if (!modelId) {
      res.status(400).json({ error: 'modelId is required.' });
      return;
    }
    const selection = selectModel(modelId);
    res.json({ ok: true, selection });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ ok: false, error: message });
  }
});

const PORT = Number(process.env.PORT) || 4000;
app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});
