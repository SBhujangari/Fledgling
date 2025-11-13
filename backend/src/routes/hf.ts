import { Router, type Request, type Response } from 'express';

import { uploadToHuggingFace, type HuggingFaceRepoType } from '../service/hfUploader';
import { deleteStoredHfToken, getStoredTokenMetadata, writeStoredHfToken } from '../service/hfTokenStore';

const router = Router();

router.get('/token', (_req: Request, res: Response) => {
  const meta = getStoredTokenMetadata();
  res.json(meta);
});

router.post('/token', (req: Request, res: Response) => {
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

router.delete('/token', (_req: Request, res: Response) => {
  deleteStoredHfToken();
  res.json({ ok: true });
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

router.post('/upload', async (req: Request<unknown, unknown, HuggingFaceUploadRequestBody>, res: Response) => {
  try {
    const body = req.body ?? {};
    const repoId = typeof body.repoId === 'string' ? body.repoId.trim() : '';
    if (!repoId) {
      res.status(400).json({ error: 'repoId is required.' });
      return;
    }

    const paths = Array.isArray(body.paths)
      ? body.paths.filter((p: unknown): p is string => typeof p === 'string' && p.trim().length > 0)
      : [];
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

export default router;
