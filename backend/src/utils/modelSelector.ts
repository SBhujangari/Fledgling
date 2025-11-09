/**
 * Automatically selects the best SLM for fine-tuning based on the task
 * In the future, this can be made more sophisticated based on:
 * - Task complexity analysis
 * - Dataset size
 * - Performance requirements
 */
export function selectSLMForAgent(taskDescription: string, originalLLM: string): string {
  // For now, use a simple default: Llama-3-8b for most tasks
  // This is a good balance of performance and cost

  // Future enhancements could include:
  // - Analyzing task complexity to choose between 1B, 3B, 7B models
  // - Matching model capabilities to task type (e.g., Phi for reasoning, Mistral for general)
  // - Considering the original LLM's capabilities

  const defaultSLM = 'llama-3-8b';

  // You can add logic here to select different models based on criteria
  // For example:
  // if (taskDescription.toLowerCase().includes('reasoning')) {
  //   return 'phi-3-mini';
  // }
  // if (originalLLM.includes('gpt-4')) {
  //   return 'llama-3-8b'; // More capable SLM for GPT-4 replacement
  // }

  return defaultSLM;
}
