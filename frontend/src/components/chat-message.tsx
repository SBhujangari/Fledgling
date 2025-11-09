import type { ChatResponse } from "@/types"

interface ChatMessageProps {
  prompt: string
  response: ChatResponse
  type: "llm" | "slm"
}

export function ChatMessage({ prompt, response, type }: ChatMessageProps) {
  const promptBg = type === "llm" ? "bg-primary/10" : "bg-accent/10"
  const responseBg = type === "llm" ? "bg-primary/5" : "bg-accent/5"
  const toolBg = type === "llm" ? "bg-primary/10" : "bg-accent/10"

  return (
    <div className="space-y-2">
      {/* User Prompt */}
      <div className={`p-4 rounded-lg ${promptBg} border border-border`}>
        <div className="text-xs font-medium text-muted-foreground mb-1">User</div>
        <div className="text-sm text-foreground">{prompt}</div>
      </div>

      {/* Model Response */}
      <div className={`p-4 rounded-lg ${responseBg} border border-border`}>
        <div className="text-xs font-medium text-muted-foreground mb-1">Assistant</div>
        <div className="text-sm text-foreground">{response.content}</div>
      </div>

      {/* Tool Calls */}
      {response.toolCalls && response.toolCalls.length > 0 && (
        <div className="pl-6 space-y-2">
          {response.toolCalls.map((toolCall, index) => (
            <div key={index} className="space-y-2">
              {/* Tool Call Input */}
              <div className={`p-3 rounded-lg ${toolBg} border border-border`}>
                <div className="text-xs font-medium text-muted-foreground mb-1">Tool Call: {toolCall.name}</div>
                <div className="text-xs font-mono text-foreground">{JSON.stringify(toolCall.input, null, 2)}</div>
              </div>

              {/* Tool Call Result */}
              <div className={`p-3 rounded-lg ${responseBg} border border-border`}>
                <div className="text-xs font-medium text-muted-foreground mb-1">Tool Result</div>
                <div className="text-xs font-mono text-foreground">{JSON.stringify(toolCall.result, null, 2)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

