import { Router } from 'express';
import { exampleWorkflows } from '../data/exampleWorkflows';

const router = Router();

router.get('/workflows', (_req, res) => {
  res.json(exampleWorkflows);
});

export default router;
