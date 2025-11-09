# AI-ATL Demo User App

Simple demo app that runs a Mastra Q&A agent with Langfuse tracing. This app validates your loader and transformer code against real Langfuse traces.

## Setup

1. **Copy `.env.example` to `.env`** and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

   Required environment variables:
   - `LANGFUSE_PUBLIC_KEY` - Your Langfuse public key
   - `LANGFUSE_SECRET_KEY` - Your Langfuse secret key
   - `LANGFUSE_BASE_URL` - Your Langfuse instance URL (e.g., http://localhost:3000 for local, https://cloud.langfuse.com for cloud)
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `LANGFUSE_AGENT_ID` - Agent identifier (e.g., `demo-agent`)

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the server**:
   ```bash
   npm run dev
   ```

   Server will run on `http://localhost:3001`

## Usage

Send a POST request to `/query` with a task:

```bash
curl -X POST http://localhost:3001/query \
  -H "Content-Type: application/json" \
  -d '{"task": "What is machine learning?"}'
```

Response:
```json
{
  "task": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
  "finishReason": "end_turn"
}
```

## Architecture

- **`src/agent.ts`** - Mastra Q&A agent definition
- **`src/index.ts`** - Express server with `/query` endpoint and Langfuse tracing

## Tracing Flow

1. Request hits `/query` endpoint
2. `startActiveObservation` creates a span named `agent-query`
3. `updateActiveTrace` sets trace metadata (agent ID, tags)
4. Agent runs and generates response
5. Langfuse automatically captures the entire trace
6. Traces appear in your Langfuse dashboard

## Next Steps

1. Run the server
2. Send a few queries
3. Check your Langfuse dashboard to see traces
4. Test your `loader.ts` and `transformer.ts` against these traces
5. Extend with more complex agent workflows
