/**
 * Model pricing constants for major providers
 * Prices are in USD per 1M tokens (input/output)
 */
export const MODEL_PRICING = {
  // OpenAI
  'gpt-4': { input: 30.0, output: 60.0 },
  'gpt-4-turbo': { input: 10.0, output: 30.0 },
  'gpt-4o': { input: 5.0, output: 15.0 },
  'gpt-3.5-turbo': { input: 0.5, output: 1.5 },

  // Anthropic
  'claude-3-opus': { input: 15.0, output: 75.0 },
  'claude-3-sonnet': { input: 3.0, output: 15.0 },
  'claude-3-haiku': { input: 0.25, output: 1.25 },
  'claude-3.5-sonnet': { input: 3.0, output: 15.0 },

  // Small/Local Models (typical costs for inference)
  'llama-3-8b': { input: 0.1, output: 0.1 },
  'mistral-7b': { input: 0.1, output: 0.1 },
  'phi-3-mini': { input: 0.05, output: 0.05 },

  // Default baseline (assumed LLM cost if not specified)
  'default-llm': { input: 10.0, output: 30.0 },
  'default-slm': { input: 0.1, output: 0.1 },
} as const;

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

export interface CostCalculationInput {
  llmModel: string; // The large model being replaced
  slmModel: string; // The small/fine-tuned model being used
  tokenUsage: TokenUsage; // Tokens processed
}

/**
 * Calculate the cost for a given model and token usage
 */
export function calculateModelCost(
  modelTag: string,
  tokenUsage: TokenUsage
): number {
  const pricing = MODEL_PRICING[modelTag as keyof typeof MODEL_PRICING] || MODEL_PRICING['default-llm'];

  // Convert from "per 1M tokens" to actual cost
  const inputCost = (tokenUsage.inputTokens / 1_000_000) * pricing.input;
  const outputCost = (tokenUsage.outputTokens / 1_000_000) * pricing.output;

  return inputCost + outputCost;
}

/**
 * Calculate cost savings by comparing LLM vs SLM inference costs
 * Returns the amount saved in USD
 */
export function calculateCostSavings(input: CostCalculationInput): number {
  const llmCost = calculateModelCost(input.llmModel, input.tokenUsage);
  const slmCost = calculateModelCost(input.slmModel, input.tokenUsage);

  const savings = llmCost - slmCost;

  // Return savings (ensure non-negative)
  return Math.max(0, savings);
}

/**
 * Calculate cost savings percentage
 */
export function calculateCostSavingsPercentage(input: CostCalculationInput): number {
  const llmCost = calculateModelCost(input.llmModel, input.tokenUsage);

  if (llmCost === 0) return 0;

  const savings = calculateCostSavings(input);
  return (savings / llmCost) * 100;
}

/**
 * Dummy function to compute model accuracy from training metrics
 * TODO: Replace with actual model evaluation logic
 * This should:
 * - Load the trained model
 * - Run evaluation on test dataset
 * - Compare against baseline LLM
 * - Return accuracy percentage
 */
export function computeModelAccuracy(
  modelPath: string,
  evaluationDataset?: unknown
): number {
  // TODO: Implement actual model accuracy computation
  // For now, return a placeholder value
  // In production, this would:
  // 1. Load model from modelPath
  // 2. Run inference on evaluation dataset
  // 3. Compare outputs with ground truth/LLM baseline
  // 4. Calculate accuracy metrics (exact match, semantic similarity, etc.)

  console.warn('computeModelAccuracy is not yet implemented, returning placeholder value');

  // Return a realistic placeholder (85-95% accuracy range)
  return 85 + Math.random() * 10;
}
