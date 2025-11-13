import { Router, type Request, type Response } from 'express';

import { getSelectedModel, listModels, selectModel } from '../service/slmCatalog';

const router = Router();

router.get('/models', (_req: Request, res: Response) => {
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

router.post('/select', (req: Request<unknown, unknown, SlmSelectionRequestBody>, res: Response) => {
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

export default router;
