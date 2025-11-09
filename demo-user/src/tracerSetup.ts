import 'dotenv/config';
import { initTracer } from '@fledgling/tracer';

initTracer({
  secretKey: process.env.LANGFUSE_SECRET_KEY!,
  publicKey: process.env.LANGFUSE_PUBLIC_KEY,
  baseUrl: process.env.LANGFUSE_BASE_URL,
  environment: process.env.NODE_ENV,
});
