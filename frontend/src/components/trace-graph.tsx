import type { TraceSample, ParsedStep } from "@/types"

interface TraceGraphProps {
  traceSample?: TraceSample
  prompt: string
}

function getNodeLabel(step: ParsedStep): string {
  if (step.type === 'generation') {
    return step.model ? `Model: ${step.model}` : 'Generation'
  }
  if (step.type === 'tool_call') {
    return step.toolName
  }
  if (step.type === 'thought') {
    return 'Thought'
  }
  return 'Step'
}

function getNodeColor(step: ParsedStep, index: number): string {
  if (step.type === 'generation') {
    return 'bg-primary'
  }
  if (step.type === 'tool_call') {
    return 'bg-accent'
  }
  if (step.type === 'thought') {
    return 'bg-muted'
  }
  return 'bg-foreground'
}

export function TraceGraph({ traceSample, prompt }: TraceGraphProps) {
  const steps = traceSample?.steps || []

  if (steps.length === 0) {
    return (
      <div className="p-6 text-center text-muted-foreground">
        <p className="text-sm">No trace data available</p>
      </div>
    )
  }

  return (
    <div className="p-6">
      <div className="flex flex-col items-center gap-4">
        {/* Start Node */}
        <div className="flex flex-col items-center gap-2">
          <div className="w-16 h-16 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center">
            <span className="text-xs font-semibold text-primary text-center px-2">Start</span>
          </div>
          <div className="text-xs text-muted-foreground max-w-[200px] text-center truncate" title={prompt}>
            {prompt}
          </div>
        </div>

        {/* Chain Line */}
        <div className="w-0.5 h-8 bg-border" />

        {/* Steps */}
        {steps.map((step, index) => (
          <div key={index} className="flex flex-col items-center gap-2">
            {/* Chain Line */}
            <div className="w-0.5 h-8 bg-border" />
            
            {/* Node */}
            <div className={`w-16 h-16 rounded-full ${getNodeColor(step, index)}/20 border-2 ${getNodeColor(step, index)} flex items-center justify-center`}>
              <span className="text-xs font-semibold text-center px-2" style={{ color: `var(--${step.type === 'generation' ? 'primary' : step.type === 'tool_call' ? 'accent' : 'muted-foreground'})` }}>
                {getNodeLabel(step)}
              </span>
            </div>
            
            {/* Node Details */}
            <div className="text-xs text-muted-foreground max-w-[200px] text-center">
              {step.type === 'tool_call' && step.toolName && (
                <div className="font-mono">{step.toolName}</div>
              )}
              {step.type === 'generation' && step.model && (
                <div className="text-[10px]">{step.model}</div>
              )}
              {step.type === 'thought' && (
                <div className="text-[10px] truncate max-w-[180px]" title={step.content}>
                  {step.content.substring(0, 30)}...
                </div>
              )}
            </div>

            {/* Chain Line */}
            {index < steps.length - 1 && (
              <div className="w-0.5 h-8 bg-border" />
            )}
          </div>
        ))}

        {/* End Node */}
        <div className="flex flex-col items-center gap-2">
          <div className="w-0.5 h-8 bg-border" />
          <div className="w-16 h-16 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center">
            <span className="text-xs font-semibold text-primary text-center px-2">End</span>
          </div>
          {traceSample?.finalResponse && (
            <div className="text-xs text-muted-foreground max-w-[200px] text-center truncate" title={traceSample.finalResponse}>
              {traceSample.finalResponse.substring(0, 40)}...
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

