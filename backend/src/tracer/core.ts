import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { setLangfuseTracerProvider } from '@langfuse/tracing';

export interface TracerConfig {
  secretKey: string;
  publicKey?: string;
  baseUrl?: string;
  environment?: string;
  release?: string;
  defaultTags?: string[];
}

let tracerProvider: NodeTracerProvider | null = null;
let tracerConfig: TracerConfig | null = null;

export function initTracer(config: TracerConfig) {
  if (tracerProvider) {
    tracerConfig = { ...tracerConfig, ...config };
    return tracerProvider;
  }

  const provider = new NodeTracerProvider({
    spanProcessors: [
      new LangfuseSpanProcessor({
        publicKey: config.publicKey,
        secretKey: config.secretKey,
        baseUrl: config.baseUrl,
        environment: config.environment,
        release: config.release,
      }),
    ],
  });
  provider.register();
  setLangfuseTracerProvider(provider);

  tracerProvider = provider;
  tracerConfig = config;
  return provider;
}

export function ensureTracerInitialized(): TracerConfig {
  if (!tracerProvider || !tracerConfig) {
    throw new Error('Tracer not initialized. Call initTracer() before using adapters.');
  }
  return tracerConfig;
}
