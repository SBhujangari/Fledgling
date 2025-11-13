import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { FinetuneSample, TracesResponse } from "@/types"

const LANGFUSE_HOST = import.meta.env.VITE_LANGFUSE_HOST || "https://cloud.langfuse.com"

function formatDate(value?: string | null) {
  if (!value) return "—"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function buildLangfuseUrl(traceId: string) {
  return `${LANGFUSE_HOST}/traces/${traceId}`
}

export function OpsTraceConsole() {
  const { data, error, isFetching, refetch, dataUpdatedAt } = useQuery<TracesResponse>({
    queryKey: ["traces-console"],
    queryFn: () => api.getTraces(),
    refetchOnWindowFocus: false,
  })

  const latestSamples = useMemo<FinetuneSample[]>(() => {
    if (!data?.samples || !Array.isArray(data.samples)) {
      return []
    }
    return data.samples.slice(0, 4)
  }, [data?.samples])

  const stats = useMemo(
    () => ({
      runs: Array.isArray(data?.runs) ? data.runs.length : 0,
      observations: Array.isArray(data?.observations) ? data.observations.length : 0,
      generations: Array.isArray(data?.generations) ? data.generations.length : 0,
      samples: latestSamples.length,
    }),
    [data?.runs, data?.observations, data?.generations, latestSamples.length],
  )

  const graphSample = latestSamples[0]
  const graphNodes = useMemo(() => {
    if (!graphSample) return []
    const firstUser = graphSample.conversation.find((msg) => msg.role === "user")
    const firstThought = graphSample.steps.find((step) => step.type === "thought")
    const firstTool = graphSample.steps.find((step) => step.type === "tool_call")
    return [
      { label: "User Prompt", content: firstUser?.content ?? "—" },
      { label: "Agent Thought", content: firstThought?.content ?? "No thought captured" },
      {
        label: "Tool Call",
        content: firstTool ? `${firstTool.toolName ?? "tool"} (${firstTool.status ?? "unknown"})` : "No tool call recorded",
      },
      { label: "Final Answer", content: graphSample.finalResponse ?? "—" },
    ]
  }, [graphSample])

  const lastUpdated = data ? formatDate(new Date(dataUpdatedAt).toISOString()) : "—"

  return (
    <section className="rounded-3xl border border-border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-semibold">Langfuse Trace Console</h2>
          <p className="text-sm text-muted-foreground">
            Live snapshot of runs, samples, and spans flowing into the fine-tune pipeline.
          </p>
          <p className="mt-1 text-xs text-muted-foreground">Last updated: {lastUpdated}</p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex items-center justify-center rounded-full border border-border px-4 py-2 text-sm font-semibold hover:bg-secondary disabled:opacity-50"
        >
          {isFetching ? "Refreshing…" : "Refresh traces"}
        </button>
      </div>

      {error instanceof Error && (
        <div className="mt-4 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          <strong>Error:</strong> {error.message}
        </div>
      )}

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(stats).map(([label, value]) => (
          <article key={label} className="rounded-2xl border border-border/80 bg-background p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="text-3xl font-semibold">{value}</p>
          </article>
        ))}
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="space-y-4 rounded-2xl border border-border/80 bg-background p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Latest Samples</h3>
            <span className="text-xs text-muted-foreground">
              {graphSample ? "Most recent four traces" : "No samples yet"}
            </span>
          </div>
          {latestSamples.length === 0 ? (
            <p className="text-sm text-muted-foreground">Trigger agent activity to populate Langfuse samples.</p>
          ) : (
            <div className="space-y-4">
              {latestSamples.map((sample) => (
                <article key={sample.traceId} className="rounded-2xl border border-border/60 bg-card p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">Agent</p>
                      <p className="font-semibold">{sample.agentId}</p>
                    </div>
                    <a
                      href={buildLangfuseUrl(sample.traceId)}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm font-semibold text-primary"
                    >
                      Open in Langfuse →
                    </a>
                  </div>
                  <p className="mt-3 text-sm leading-relaxed">{sample.finalResponse || "Awaiting final response."}</p>
                  <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
                    <span>Steps: {sample.steps.length}</span>
                    <span>Turns: {sample.conversation.length}</span>
                    <span>Trace: {sample.traceId.slice(0, 8)}…</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-5 rounded-2xl border border-border/80 bg-background p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Agent Graph</h3>
            {graphSample && (
              <span className="text-xs text-muted-foreground">Trace {graphSample.traceId.slice(0, 8)}…</span>
            )}
          </div>
          {graphSample ? (
            <div className="flex flex-col gap-4">
              {graphNodes.map((node, index) => (
                <div key={node.label} className="flex items-center gap-3">
                  <div className="flex-1 rounded-2xl border border-border/70 bg-card p-4">
                    <p className="text-xs uppercase text-muted-foreground">{node.label}</p>
                    <p className="mt-2 whitespace-pre-wrap text-sm">{node.content}</p>
                  </div>
                  {index < graphNodes.length - 1 && <span className="text-2xl text-muted-foreground">→</span>}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Run a trace to visualize the agent flow.</p>
          )}
        </div>
      </div>

      {data && (
        <details className="mt-8 rounded-2xl border border-border/60 bg-background p-4">
          <summary className="cursor-pointer text-sm font-semibold">Raw JSON payload</summary>
          <pre className="mt-4 max-h-[400px] overflow-auto rounded-xl bg-muted/40 p-4 text-xs">
            {JSON.stringify(data, null, 2)}
          </pre>
        </details>
      )}
    </section>
  )
}
