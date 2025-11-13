import { useState } from "react"

interface TraceObservation {
  id: string
  type: string
  name: string
  startTime: string
  input?: any
  output?: any
  model?: string
  metadata?: Record<string, any>
}

interface TraceData {
  id: string
  name: string
  timestamp: string
  metadata?: Record<string, any>
  input?: any[]
  output?: any
  observations?: TraceObservation[]
}

interface TraceVisualizationProps {
  trace: TraceData
}

export function TraceVisualization({ trace }: TraceVisualizationProps) {
  const [selectedObservation, setSelectedObservation] = useState<TraceObservation | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(["flow", "timeline"]))

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(section)) {
        newSet.delete(section)
      } else {
        newSet.add(section)
      }
      return newSet
    })
  }

  const observations = trace.observations || []
  const generations = observations.filter((obs) => obs.type === "generation")
  const toolCalls = observations.filter((obs) => obs.type === "tool")

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString()
    } catch {
      return timestamp
    }
  }

  const formatJSON = (data: any) => {
    if (typeof data === "string") return data
    return JSON.stringify(data, null, 2)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-border bg-card p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-semibold">{trace.name}</h2>
              <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 dark:bg-green-900 dark:text-green-300">
                Completed
              </span>
            </div>
            <p className="font-mono text-sm text-muted-foreground">{trace.id}</p>
            <p className="text-xs text-muted-foreground">{formatTimestamp(trace.timestamp)}</p>
          </div>
          {trace.metadata && (
            <div className="rounded-lg bg-muted px-3 py-2 text-xs">
              <p className="font-semibold text-muted-foreground">Metadata</p>
              {Object.entries(trace.metadata).map(([key, value]) => (
                <p key={key} className="mt-1">
                  <span className="font-medium">{key}:</span> {String(value)}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Agent Flow Diagram */}
      <section className="rounded-2xl border border-border bg-card p-6">
        <button
          onClick={() => toggleSection("flow")}
          className="flex w-full items-center justify-between text-left"
        >
          <h3 className="text-xl font-semibold">Agent Execution Flow</h3>
          <span className="text-muted-foreground">{expandedSections.has("flow") ? "−" : "+"}</span>
        </button>
        {expandedSections.has("flow") && (
          <div className="mt-6">
            <div className="flex flex-wrap items-center gap-4">
              {/* Input Node */}
              <div className="flex-shrink-0 rounded-xl border border-blue-500/40 bg-blue-50 px-4 py-3 dark:bg-blue-950/30">
                <p className="text-xs font-semibold text-blue-600 dark:text-blue-400">INPUT</p>
                <p className="mt-1 text-sm">User Query</p>
              </div>

              <span className="text-xl text-muted-foreground">→</span>

              {/* Generation Nodes */}
              {generations.map((gen, index) => (
                <div key={gen.id} className="flex items-center gap-4">
                  <button
                    onClick={() => setSelectedObservation(gen)}
                    className={`flex-shrink-0 rounded-xl border px-4 py-3 transition-all ${
                      selectedObservation?.id === gen.id
                        ? "border-purple-500 bg-purple-100 dark:bg-purple-950/50"
                        : "border-border bg-background hover:bg-muted"
                    }`}
                  >
                    <p className="text-xs font-semibold text-purple-600 dark:text-purple-400">
                      GENERATION {index + 1}
                    </p>
                    <p className="mt-1 text-sm">{gen.name}</p>
                    {gen.model && <p className="mt-1 text-xs text-muted-foreground">{gen.model}</p>}
                  </button>
                  {index < generations.length - 1 && <span className="text-xl text-muted-foreground">→</span>}
                </div>
              ))}

              {toolCalls.length > 0 && (
                <>
                  <span className="text-xl text-muted-foreground">→</span>
                  <div className="flex items-center gap-4">
                    {toolCalls.map((tool, index) => (
                      <div key={tool.id} className="flex items-center gap-4">
                        <button
                          onClick={() => setSelectedObservation(tool)}
                          className={`flex-shrink-0 rounded-xl border px-4 py-3 transition-all ${
                            selectedObservation?.id === tool.id
                              ? "border-amber-500 bg-amber-100 dark:bg-amber-950/50"
                              : "border-border bg-background hover:bg-muted"
                          }`}
                        >
                          <p className="text-xs font-semibold text-amber-600 dark:text-amber-400">TOOL CALL</p>
                          <p className="mt-1 text-sm">{tool.name}</p>
                        </button>
                        {index < toolCalls.length - 1 && <span className="text-xl text-muted-foreground">→</span>}
                      </div>
                    ))}
                  </div>
                </>
              )}

              <span className="text-xl text-muted-foreground">→</span>

              {/* Output Node */}
              <div className="flex-shrink-0 rounded-xl border border-green-500/40 bg-green-50 px-4 py-3 dark:bg-green-950/30">
                <p className="text-xs font-semibold text-green-600 dark:text-green-400">OUTPUT</p>
                <p className="mt-1 text-sm">Final Response</p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Execution Timeline */}
      <section className="rounded-2xl border border-border bg-card p-6">
        <button
          onClick={() => toggleSection("timeline")}
          className="flex w-full items-center justify-between text-left"
        >
          <h3 className="text-xl font-semibold">Execution Timeline</h3>
          <span className="text-muted-foreground">{expandedSections.has("timeline") ? "−" : "+"}</span>
        </button>
        {expandedSections.has("timeline") && (
          <div className="mt-6 space-y-3">
            {observations.map((obs, index) => (
              <button
                key={obs.id}
                onClick={() => setSelectedObservation(obs)}
                className={`w-full rounded-lg border p-4 text-left transition-all ${
                  selectedObservation?.id === obs.id
                    ? "border-primary bg-primary/5"
                    : "border-border bg-background hover:bg-muted"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                      {index + 1}
                    </span>
                    <div>
                      <p className="font-semibold">{obs.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {obs.type} • {formatTimestamp(obs.startTime)}
                      </p>
                    </div>
                  </div>
                  {obs.model && (
                    <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium">{obs.model}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      {/* Input/Output Details */}
      <section className="rounded-2xl border border-border bg-card p-6">
        <button
          onClick={() => toggleSection("io")}
          className="flex w-full items-center justify-between text-left"
        >
          <h3 className="text-xl font-semibold">Input & Output</h3>
          <span className="text-muted-foreground">{expandedSections.has("io") ? "−" : "+"}</span>
        </button>
        {expandedSections.has("io") && (
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            {/* Trace Input */}
            <div className="rounded-lg border border-border bg-background p-4">
              <p className="text-sm font-semibold text-muted-foreground">Trace Input</p>
              <pre className="mt-3 max-h-[300px] overflow-auto rounded-lg bg-muted p-3 text-xs">
                {formatJSON(trace.input)}
              </pre>
            </div>

            {/* Trace Output */}
            <div className="rounded-lg border border-border bg-background p-4">
              <p className="text-sm font-semibold text-muted-foreground">Trace Output</p>
              <pre className="mt-3 max-h-[300px] overflow-auto rounded-lg bg-muted p-3 text-xs">
                {formatJSON(trace.output)}
              </pre>
            </div>
          </div>
        )}
      </section>

      {/* Selected Observation Details */}
      {selectedObservation && (
        <section className="rounded-2xl border border-primary/40 bg-primary/5 p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xl font-semibold">Selected: {selectedObservation.name}</h3>
            <button
              onClick={() => setSelectedObservation(null)}
              className="rounded-lg px-3 py-1 text-sm hover:bg-background"
            >
              ✕ Close
            </button>
          </div>
          <div className="space-y-4">
            <div>
              <p className="text-sm font-semibold text-muted-foreground">Type</p>
              <p className="mt-1">{selectedObservation.type}</p>
            </div>
            <div>
              <p className="text-sm font-semibold text-muted-foreground">ID</p>
              <p className="mt-1 font-mono text-xs">{selectedObservation.id}</p>
            </div>
            {selectedObservation.model && (
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Model</p>
                <p className="mt-1">{selectedObservation.model}</p>
              </div>
            )}
            {selectedObservation.input && (
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Input</p>
                <pre className="mt-2 max-h-[200px] overflow-auto rounded-lg border border-border bg-background p-3 text-xs">
                  {formatJSON(selectedObservation.input)}
                </pre>
              </div>
            )}
            {selectedObservation.output && (
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Output</p>
                <pre className="mt-2 max-h-[200px] overflow-auto rounded-lg border border-border bg-background p-3 text-xs">
                  {formatJSON(selectedObservation.output)}
                </pre>
              </div>
            )}
            {selectedObservation.metadata && (
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Metadata</p>
                <pre className="mt-2 max-h-[150px] overflow-auto rounded-lg border border-border bg-background p-3 text-xs">
                  {formatJSON(selectedObservation.metadata)}
                </pre>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Summary Statistics */}
      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Total Steps</p>
          <p className="mt-2 text-3xl font-semibold">{observations.length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Generations</p>
          <p className="mt-2 text-3xl font-semibold">{generations.length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Tool Calls</p>
          <p className="mt-2 text-3xl font-semibold">{toolCalls.length}</p>
        </div>
      </section>
    </div>
  )
}
