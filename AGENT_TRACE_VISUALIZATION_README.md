# Agent Trace Visualization - Self-Hosted Implementation

## Overview

This implementation provides a **self-hosted agent trace visualization system** directly embedded in the Agent Trace Console Page. No Langfuse API key required!

## Features Implemented

### 1. **Backend API Endpoints** ([backend/src/routes/traces.ts](backend/src/routes/traces.ts))

Added two new endpoints to serve local trace data:

- `GET /api/traces/local` - Retrieves all traces from local storage
- `GET /api/traces/local/:traceId` - Retrieves a specific trace by ID

These endpoints read from `storage/dummy_langfuse_traces.jsonl` and work without any external API keys.

### 2. **Trace Visualization Component** ([frontend/src/components/TraceVisualization.tsx](frontend/src/components/TraceVisualization.tsx))

A comprehensive React component that displays:

#### **Agent Execution Flow Diagram**
- Visual representation of the agent execution pipeline
- Shows INPUT → GENERATION → TOOL CALLS → OUTPUT flow
- Interactive nodes that can be clicked to view details
- Color-coded by type (blue for input, purple for generations, amber for tools, green for output)

#### **Execution Timeline**
- Step-by-step chronological view of all observations
- Shows generation steps and tool calls in order
- Displays timing and model information
- Clickable items to view detailed information

#### **Input/Output Details**
- Side-by-side view of trace input and output
- Formatted JSON display with syntax highlighting
- Collapsible section for clean UI

#### **Selected Observation Details**
- Detailed view when clicking on any step
- Shows full input, output, metadata, and model information
- Easy-to-read formatted JSON

#### **Summary Statistics**
- Total steps count
- Number of generations
- Number of tool calls

### 3. **Integration into TracesPage** ([frontend/src/pages/TracesPage.tsx](frontend/src/pages/TracesPage.tsx))

Added a new section "Local Agent Traces (Self-Hosted)" that:

- Displays a grid of available traces
- Auto-refreshes every 10 seconds to show new traces
- Click any trace card to view full visualization
- "Back to trace list" button for easy navigation
- Clear indicator that no API key is required

## How It Works

### Data Flow

```
Local Trace File (storage/dummy_langfuse_traces.jsonl)
    ↓
Backend API (/api/traces/local)
    ↓
Frontend API Client (api.getLocalTraces())
    ↓
TracesPage (displays trace cards)
    ↓
TraceVisualization Component (full interactive view)
```

### Trace Data Format

The system expects traces in this format:

```json
{
  "id": "trace-id",
  "name": "trace-name",
  "timestamp": "2025-11-12T19:17:14.567672Z",
  "metadata": {
    "agent_id": "agent-name",
    "source": "workflow_source"
  },
  "input": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User query"}
  ],
  "output": "Final response or JSON output",
  "observations": [
    {
      "id": "obs-id",
      "type": "generation",
      "name": "generation-name",
      "startTime": "2025-11-12T19:17:14.567672Z",
      "input": "Input to this step",
      "output": "Output from this step",
      "model": "model-name"
    },
    {
      "id": "tool-id",
      "type": "tool",
      "name": "tool_name",
      "startTime": "2025-11-12T19:17:14.567672Z",
      "input": {"param": "value"},
      "output": {"status": "success"},
      "metadata": {"category": "tool"}
    }
  ]
}
```

## Usage

### Viewing Traces

1. Navigate to the Agent Trace Console page
2. Scroll to the "Local Agent Traces (Self-Hosted)" section
3. Click on any trace card to view the full visualization
4. Explore the different sections:
   - Agent Execution Flow (click nodes to see details)
   - Execution Timeline (click steps to see details)
   - Input/Output (view formatted JSON)
   - Summary Statistics

### Adding New Traces

Simply add trace JSON objects (one per line) to:
```
storage/dummy_langfuse_traces.jsonl
```

The UI will automatically pick them up within 10 seconds (auto-refresh).

## Comparison to Langfuse

### Traditional Langfuse Approach
- ❌ Requires cloud API key
- ❌ External dependency
- ❌ Data sent to third-party
- ❌ Requires network connection
- ✅ Hosted UI

### Our Self-Hosted Approach
- ✅ No API key required
- ✅ Fully local/self-hosted
- ✅ Data stays on your server
- ✅ Works offline
- ✅ Embedded in your application
- ✅ Full customization control

## Key Benefits

1. **Privacy**: All trace data stays on your infrastructure
2. **Cost**: No API charges or usage limits
3. **Speed**: Local file access is faster than API calls
4. **Reliability**: No external service dependencies
5. **Control**: Full customization of visualization

## Technical Details

### Backend
- Express.js endpoints
- Reads from local `.jsonl` file
- No external API calls
- TypeScript with full type safety

### Frontend
- React with TypeScript
- TanStack Query for data fetching
- Auto-refresh every 10 seconds
- Responsive design with Tailwind CSS
- Interactive flow diagrams
- Collapsible sections for clean UI

## Files Modified

1. **Backend**
   - [backend/src/routes/traces.ts](backend/src/routes/traces.ts) - Added `/api/traces/local` endpoints

2. **Frontend**
   - [frontend/src/components/TraceVisualization.tsx](frontend/src/components/TraceVisualization.tsx) - New visualization component
   - [frontend/src/pages/TracesPage.tsx](frontend/src/pages/TracesPage.tsx) - Integrated visualization
   - [frontend/src/lib/api.ts](frontend/src/lib/api.ts) - Added API client methods

## Future Enhancements

Potential improvements:

1. **Performance Metrics**: Add latency charts and cost analysis
2. **Filtering**: Filter traces by agent, date, status
3. **Search**: Search within trace content
4. **Export**: Download traces as JSON or CSV
5. **Comparison**: Side-by-side trace comparison
6. **Real-time Updates**: WebSocket support for live traces
7. **Trace Recording**: Built-in agent instrumentation

## Examples

The system comes with sample traces in `storage/dummy_langfuse_traces.jsonl` showing:
- Medical appointment scheduling
- Hotel booking
- Restaurant reservations
- Courier pickup scheduling
- Finance report generation
- Branch office registration

These demonstrate the full capabilities of the visualization system.

---

**Status**: ✅ Complete and Ready to Use

No configuration needed - just navigate to the Agent Trace Console page!
