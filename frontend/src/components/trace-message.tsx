interface ToolCall {
  name: string
  input: any
  result: any
}

interface ChatResponse {
  content: string
  toolCalls?: ToolCall[]
}

interface TraceMessageProps {
  prompt: string
  response: ChatResponse
  type: "llm" | "slm"
}

export function TraceMessage({ prompt, response, type }: TraceMessageProps) {
  const borderColor = type === "llm" ? "border-primary/30" : "border-accent/30"
  const textColor = type === "llm" ? "text-primary" : "text-accent"

  return (
    <div className="space-y-3">
      {/* Trace Entry - Prompt */}
      <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${type === "llm" ? "bg-primary" : "bg-accent"}`} />
          <span className="text-xs font-mono text-muted-foreground">TRACE_START</span>
          <span className="text-xs text-muted-foreground">0ms</span>
        </div>
        <div className="text-xs font-mono text-foreground pl-4">{prompt}</div>
      </div>

      {/* Trace Entry - Model Processing */}
      <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2 mb-2">
          <div className={`size-2 rounded-full ${type === "llm" ? "bg-primary" : "bg-accent"}`} />
          <span className="text-xs font-mono text-muted-foreground">MODEL_PROCESS</span>
          <span className="text-xs text-muted-foreground">45ms</span>
        </div>
        <div className="text-xs font-mono text-foreground pl-4">{response.content}</div>
      </div>

      {/* Trace Entry - Tool Calls */}
      {response.toolCalls && response.toolCalls.length > 0 && (
        <div className="pl-4 space-y-3">
          {response.toolCalls.map((toolCall, index) => (
            <div key={index} className="space-y-3">
              <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`size-2 rounded-full ${type === "llm" ? "bg-primary" : "bg-accent"}`} />
                  <span className="text-xs font-mono text-muted-foreground">TOOL_CALL</span>
                  <span className={`text-xs font-mono ${textColor}`}>{toolCall.name}</span>
                  <span className="text-xs text-muted-foreground">{120 + index * 30}ms</span>
                </div>
                <div className="text-xs font-mono text-foreground pl-4">{JSON.stringify(toolCall.input, null, 2)}</div>
              </div>

              <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`size-2 rounded-full ${type === "llm" ? "bg-primary" : "bg-accent"}`} />
                  <span className="text-xs font-mono text-muted-foreground">TOOL_RESULT</span>
                  <span className="text-xs text-muted-foreground">{180 + index * 30}ms</span>
                </div>
                <div className="text-xs font-mono text-foreground pl-4">{JSON.stringify(toolCall.result, null, 2)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Trace Entry - Complete */}
      <div className={`p-3 rounded-lg border ${borderColor} bg-background/50`}>
        <div className="flex items-center gap-2">
          <div className={`size-2 rounded-full ${type === "llm" ? "bg-primary" : "bg-accent"}`} />
          <span className="text-xs font-mono text-muted-foreground">TRACE_END</span>
          <span className="text-xs text-muted-foreground">245ms</span>
        </div>
      </div>
    </div>
  )
}

