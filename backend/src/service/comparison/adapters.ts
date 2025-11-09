import { AgentExecutionPayload, AgentExecutionResult, ModelEndpointConfig } from './types';

export interface AgentComparisonAdapter {
  name: string;
  supports(config: ModelEndpointConfig): boolean;
  run(config: ModelEndpointConfig, payload: AgentExecutionPayload): Promise<AgentExecutionResult>;
}
