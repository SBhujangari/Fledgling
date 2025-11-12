import { Router, Request, Response } from 'express';

const router = Router();

const OPENAI_MODELS = [
  { id: 'openai/gpt-5', label: 'GPT-5', description: 'Highest quality GPT-5 model', provider: 'openai' },
  { id: 'openai/gpt-5-mini', label: 'GPT-5 Mini', description: 'Balanced GPT-5 variant', provider: 'openai' },
  { id: 'openai/gpt-5-nano', label: 'GPT-5 Nano', description: 'Cost-effective GPT-5 tier', provider: 'openai' },
  { id: 'openai/gpt-4o-mini', label: 'GPT-4o Mini', description: 'Fast GPT-4o variant', provider: 'openai' },
];

router.get('/llm', (_req: Request, res: Response) => {
  res.json(OPENAI_MODELS);
});

export default router;
