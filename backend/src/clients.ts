import { createOpenAI } from '@ai-sdk/openai';

export const openrouterClient = createOpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY || '',
});

export const openaiClient = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
});
