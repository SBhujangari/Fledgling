import type { ChatResponse, TraceSample, ParsedStep } from "@/types"

interface TraceMessageProps {
  prompt: string
  response: ChatResponse
  type: "llm" | "slm"
  traceSample?: TraceSample
}

function formatTimestamp(timestamp?: string): string {
  if (!timestamp) return "N/A"
  try {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 })
  } catch {
    return timestamp
  }
}

function calculateDuration(start?: string, end?: string): string {
  if (!start || !end) return "N/A"
  try {
    const startTime = new Date(start).getTime()
    const endTime = new Date(end).getTime()
    const duration = endTime - startTime
    return `${duration}ms`
  } catch {
    return "N/A"
  }
}

function stringifyContent(content: unknown): string {
  if (typeof content === 'string') return content
  if (content === null || content === undefined) return ''
  try {
    return JSON.stringify(content, null, 2)
  } catch {
    return String(content)
  }
}

function renderStep(step: ParsedStep, index: number, borderColor: string, dotColor: string, textColor: string) {
  if (step.type === 'generation') {
    return (
      <div key={index} className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${dotColor}`} />
          <span className="text-xs font-mono text-muted-foreground">GENERATION</span>
          {step.model && (
            <span className="text-xs font-mono text-foreground">{step.model}</span>
          )}
          {step.usage && (
            <span className="text-xs text-muted-foreground">
              {step.usage.totalTokens ? `${step.usage.totalTokens} tokens` : ''}
            </span>
          )}
        </div>
        {step.prompt && (
          <div className="mb-2">
            <div className="text-xs text-muted-foreground mb-1">Prompt:</div>
            <div className="text-xs font-mono text-foreground pl-4">{stringifyContent(step.prompt)}</div>
          </div>
        )}
        {step.completion && (
          <div>
            <div className="text-xs text-muted-foreground mb-1">Completion:</div>
            <div className="text-xs font-mono text-foreground pl-4">{stringifyContent(step.completion)}</div>
          </div>
        )}
      </div>
    )
  }

  if (step.type === 'tool_call') {
    return (
      <div key={index} className="space-y-2">
        <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${dotColor}`} />
          <span className="text-xs font-mono text-muted-foreground">TOOL_CALL</span>
            <span className={`text-xs font-mono ${textColor}`}>{step.toolName}</span>
            {step.startedAt && (
              <span className="text-xs text-muted-foreground">{formatTimestamp(step.startedAt)}</span>
            )}
            {step.status && (
              <span className={`text-xs ${step.status === 'success' ? 'text-green-500' : 'text-red-500'}`}>
                {step.status}
              </span>
            )}
          </div>
          <div className="mb-2">
            <div className="text-xs text-muted-foreground mb-1">Input:</div>
            <div className="text-xs font-mono text-foreground pl-4">{stringifyContent(step.input)}</div>
          </div>
          {step.output !== undefined && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">Output:</div>
              <div className="text-xs font-mono text-foreground pl-4">{stringifyContent(step.output)}</div>
            </div>
          )}
          {step.startedAt && step.completedAt && (
            <div className="mt-2 text-xs text-muted-foreground">
              Duration: {calculateDuration(step.startedAt, step.completedAt)}
            </div>
          )}
        </div>
      </div>
    )
  }

  if (step.type === 'thought') {
    return (
      <div key={index} className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${dotColor}`} />
          <span className="text-xs font-mono text-muted-foreground">THOUGHT</span>
          {step.timestamp && (
            <span className="text-xs text-muted-foreground">{formatTimestamp(step.timestamp)}</span>
          )}
        </div>
        <div className="text-xs font-mono text-foreground pl-4">{step.content}</div>
      </div>
    )
  }

  return null
}

export function TraceMessage({ prompt, response, type, traceSample }: TraceMessageProps) {
  const borderColor = type === "llm" ? "border-primary/30" : "border-accent/30"
  const textColor = type === "llm" ? "text-primary" : "text-accent"
  const dotColor = type === "llm" ? "bg-primary" : "bg-accent"

  // For SLM, use traceSample if available; for LLM, show simple trace
  const steps = type === "slm" && traceSample?.steps ? traceSample.steps : null

  return (
    <div className="space-y-3">
      {/* Trace Start - User Prompt */}
      <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${dotColor}`} />
          <span className="text-xs font-mono text-muted-foreground">TRACE_START</span>
          <span className="text-xs text-muted-foreground">User Prompt</span>
        </div>
        <div className="text-xs font-mono text-foreground pl-4">{prompt}</div>
      </div>

      {/* Trace Steps from traceSample */}
      {steps && steps.length > 0 ? (
        <div className="space-y-3">
          {steps.map((step, index) => renderStep(step, index, borderColor, dotColor, textColor))}
        </div>
      ) : (
        // Fallback: Show simple generation step if no traceSample
        <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
          <div className="flex items-center gap-2 mb-2">
            <div className={`size-2 rounded-full ${dotColor}`} />
            <span className="text-xs font-mono text-muted-foreground">GENERATION</span>
          </div>
          <div className="text-xs font-mono text-foreground pl-4">{response.content}</div>
        </div>
      )}

      {/* Tool Calls from response (fallback if not in traceSample) */}
      {!steps && response.toolCalls && response.toolCalls.length > 0 && (
        <div className="space-y-3">
          {response.toolCalls.map((toolCall, index) => (
            <div key={index} className="space-y-2">
              <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`size-2 rounded-full ${dotColor}`} />
                  <span className="text-xs font-mono text-muted-foreground">TOOL_CALL</span>
                  <span className={`text-xs font-mono ${textColor}`}>{toolCall.name}</span>
                </div>
                <div className="text-xs font-mono text-foreground pl-4">{JSON.stringify(toolCall.input, null, 2)}</div>
              </div>
              <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`size-2 rounded-full ${dotColor}`} />
                  <span className="text-xs font-mono text-muted-foreground">TOOL_RESULT</span>
                </div>
                <div className="text-xs font-mono text-foreground pl-4">{JSON.stringify(toolCall.result, null, 2)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Trace End */}
      <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2">
          <div className={`size-2 rounded-full ${dotColor}`} />
          <span className="text-xs font-mono text-muted-foreground">TRACE_END</span>
          {traceSample?.finalResponse && (
            <span className="text-xs text-muted-foreground ml-2">Final Response</span>
          )}
        </div>
      </div>

      {/* Usage Stats */}
      {traceSample?.usage && (
        <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
          <div className="text-xs font-mono text-muted-foreground mb-2">Usage Stats</div>
          <div className="text-xs text-foreground pl-4 space-y-1">
            {traceSample.usage.inputTokens && (
              <div>Input Tokens: {traceSample.usage.inputTokens}</div>
            )}
            {traceSample.usage.outputTokens && (
              <div>Output Tokens: {traceSample.usage.outputTokens}</div>
            )}
            {traceSample.usage.totalTokens && (
              <div>Total Tokens: {traceSample.usage.totalTokens}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
