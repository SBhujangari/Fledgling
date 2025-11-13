import { useMemo, useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Navigation } from "@/components/navigation"
import { api } from "@/lib/api"
import type { ConversationMessage, ExampleWorkflow, FinetuneSample, TracesResponse } from "@/types"

const LANGFUSE_HOST = import.meta.env.VITE_LANGFUSE_HOST || "https://cloud.langfuse.com"

function formatDate(value?: string | null) {
  if (!value) return "—"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function buildLangfuseUrl(traceId: string) {
  return `${LANGFUSE_HOST}/traces/${traceId}`
}

interface AgentFormState {
  id: string
  name: string
  taskDescription: string
  instructions: string
  originalLLM: string
  tags: string
}

const defaultAgentForm: AgentFormState = {
  id: "",
  name: "",
  taskDescription: "",
  instructions: "",
  originalLLM: "gpt-4o-mini",
  tags: "client,imported",
}

export default function TracesPage() {
  const [agentForm, setAgentForm] = useState<AgentFormState>(defaultAgentForm)
  const [registerMessage, setRegisterMessage] = useState<string | null>(null)

  const { data, error, isLoading, refetch, dataUpdatedAt } = useQuery({
    queryKey: ["traces-console"],
    queryFn: () => api.getTraces(),
    refetchOnWindowFocus: false,
  })

  const registerAgentMutation = useMutation({
    mutationFn: async (payload: AgentFormState) => {
      const tags = payload.tags
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean)
      return api.registerAgent({
        id: payload.id,
        name: payload.name || payload.id,
        taskDescription: payload.taskDescription,
        instructions: payload.instructions,
        originalLLM: payload.originalLLM,
        tags,
      })
    },
    onSuccess: () => {
      setRegisterMessage("Agent registered and ready for tracing.")
      setAgentForm(defaultAgentForm)
    },
  })

  const triggerTrainMutation = useMutation({
    mutationFn: api.train,
    onSuccess: () => {
      setRegisterMessage("Fine-tune pipeline triggered. Monitor Langfuse for progress.")
    },
  })

  const { data: exampleWorkflows } = useQuery<ExampleWorkflow[]>({
    queryKey: ["example-workflows"],
    queryFn: () => api.getExampleWorkflows(),
  })

  const latestSamples: FinetuneSample[] = useMemo(() => data?.samples?.slice?.(0, 4) ?? [], [data?.samples])
  const stats = useMemo(
    () => ({
      runs: data?.runs?.length ?? 0,
      observations: data?.observations?.length ?? 0,
      generations: data?.generations?.length ?? 0,
      samples: latestSamples.length,
    }),
    [data?.runs, data?.observations, data?.generations, latestSamples.length],
  )

  const graphSample = latestSamples[0]
  const graphNodes = useMemo(() => {
    if (!graphSample) return []
    const firstUser = graphSample.conversation.find((msg: ConversationMessage) => msg.role === "user")
    const firstThought = graphSample.steps.find((step) => step.type === "thought")
    const firstTool = graphSample.steps.find((step) => step.type === "tool_call")
    return [
      { label: "User Prompt", content: firstUser?.content ?? "—" },
      { label: "Agent Thought", content: firstThought?.content ?? "No reasoning captured" },
      {
        label: "Tool Call",
        content: firstTool ? `${firstTool.toolName ?? "tool"} (${firstTool.status ?? "unknown"})` : "No tool call recorded",
      },
      { label: "Final Answer", content: graphSample.finalResponse ?? "—" },
    ]
  }, [graphSample])

  return (
    <div className="min-h-screen w-full bg-background text-foreground">
      <Navigation />
      <div className="w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <header className="space-y-2">
          <p className="text-sm text-muted-foreground">Langfuse Telemetry</p>
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Agent Trace Console</h1>
              <p className="text-muted-foreground">
                Inspect captured traces, register new client workflows, and trigger the automated fine-tuning loop.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">
                Last updated <strong>{data ? formatDate(new Date(dataUpdatedAt).toISOString()) : "—"}</strong>
              </span>
              <button
                type="button"
                onClick={() => refetch()}
                disabled={isLoading}
                className="rounded-full px-4 py-2 border border-border text-sm font-medium hover:bg-muted disabled:opacity-50"
              >
                {isLoading ? "Refreshing…" : "Refresh traces"}
              </button>
            </div>
          </div>
        </header>

        {error instanceof Error && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            <strong>Error:</strong> {error.message}
          </div>
        )}

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Object.entries(stats).map(([label, value]) => (
            <article key={label} className="rounded-xl border border-border bg-card p-4 shadow-sm">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
              <p className="text-3xl font-semibold">{value}</p>
            </article>
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-5">
          <div className="rounded-2xl border border-border bg-card p-5 lg:col-span-3 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold">Latest samples</h2>
                <p className="text-sm text-muted-foreground">Most recent Langfuse traces prepared for the pipeline</p>
              </div>
            </div>
            {latestSamples.length === 0 ? (
              <p className="text-muted-foreground text-sm">Run an agent to produce Langfuse samples.</p>
            ) : (
              <div className="space-y-4">
                {latestSamples.map((sample) => (
                  <article key={sample.traceId} className="rounded-2xl border border-border/80 p-4 space-y-3 bg-background">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground text-xs uppercase">Agent</p>
                        <p className="font-semibold">{sample.agentId}</p>
                      </div>
                      <a
                        href={buildLangfuseUrl(sample.traceId)}
                        target="_blank"
                        rel="noreferrer"
                        className="text-primary text-sm font-medium"
                      >
                        View in Langfuse →
                      </a>
                    </div>
                    <p className="text-sm leading-relaxed">{sample.finalResponse || "No final response recorded yet."}</p>
                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                      <span>Steps: {sample.steps.length}</span>
                      <span>Conversation turns: {sample.conversation.length}</span>
                      <span>Trace: {sample.traceId.slice(0, 8)}…</span>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-border bg-card p-5 space-y-4 lg:col-span-2">
            <div>
              <h2 className="text-xl font-semibold">Import client workflow</h2>
              <p className="text-sm text-muted-foreground">
                Register the agent + instructions; tracing starts immediately so we can auto-build datasets.
              </p>
            </div>
            <form
              className="space-y-3"
              onSubmit={(event) => {
                event.preventDefault()
                setRegisterMessage(null)
                registerAgentMutation.mutate(agentForm)
              }}
            >
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Agent ID</label>
                <input
                  required
                  value={agentForm.id}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, id: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  placeholder="acme-support-agent"
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Display name</label>
                <input
                  value={agentForm.name}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, name: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  placeholder="Acme Support Agent"
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Task description</label>
                <textarea
                  value={agentForm.taskDescription}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, taskDescription: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm min-h-16"
                  placeholder="Summarizes onboarding tickets..."
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Instructions / system prompt</label>
                <textarea
                  value={agentForm.instructions}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, instructions: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm min-h-24"
                  placeholder="You are a helpful support agent..."
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Original LLM deployment</label>
                <input
                  value={agentForm.originalLLM}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, originalLLM: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  placeholder="gpt-4o-mini"
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Tags (comma separated)</label>
                <input
                  value={agentForm.tags}
                  onChange={(event) => setAgentForm((prev) => ({ ...prev, tags: event.target.value }))}
                  className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  placeholder="client,imported"
                />
              </div>
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button
                  type="submit"
                  disabled={registerAgentMutation.isPending}
                  className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
                >
                  {registerAgentMutation.isPending ? "Registering…" : "Register workflow"}
                </button>
                <button
                  type="button"
                  disabled={triggerTrainMutation.isPending}
                  onClick={() => triggerTrainMutation.mutate()}
                  className="rounded-full border border-border px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  {triggerTrainMutation.isPending ? "Starting…" : "Kick off fine-tune"}
                </button>
              </div>
              {registerMessage && <p className="text-sm text-muted-foreground">{registerMessage}</p>}
              {registerAgentMutation.error instanceof Error && (
                <p className="text-sm text-destructive">{registerAgentMutation.error.message}</p>
              )}
            </form>
          </div>
        </section>

        {exampleWorkflows && exampleWorkflows.length > 0 && (
          <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-semibold">Cookbook example workflows</h2>
                <p className="text-sm text-muted-foreground">
                  LangGraph + dataset recipes sourced from the Langfuse cookbook, wired to our evaluation datasets.
                </p>
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {exampleWorkflows.map((workflow) => (
                <article key={workflow.id} className="rounded-2xl border border-border bg-card p-5 space-y-3">
                  <div className="space-y-1">
                    <h3 className="text-lg font-semibold">{workflow.title}</h3>
                    <p className="text-sm text-muted-foreground">{workflow.summary}</p>
                  </div>
                  <div className="text-xs font-mono bg-muted/60 rounded-xl px-3 py-2">
                    Dataset: <span className="text-foreground">{workflow.datasetPath}</span>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs uppercase text-muted-foreground">Features</p>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                      {workflow.features.map((ft) => (
                        <li key={ft}>{ft}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs uppercase text-muted-foreground">LangGraph snippet</p>
                    <pre className="max-h-[220px] overflow-auto rounded-xl border border-border/70 bg-background px-3 py-2 text-xs">
                      <code>{workflow.codeSnippet}</code>
                    </pre>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      className="rounded-full border border-border px-3 py-1.5 text-xs font-medium"
                      onClick={async () => {
                        await navigator.clipboard.writeText(workflow.evaluationCommand)
                      }}
                    >
                      Copy eval command
                    </button>
                    <button
                      type="button"
                      className="rounded-full border border-border px-3 py-1.5 text-xs font-medium"
                      onClick={() =>
                        setAgentForm({
                          id: workflow.defaultAgent.id,
                          name: workflow.defaultAgent.name,
                          taskDescription: workflow.defaultAgent.taskDescription,
                          instructions: workflow.defaultAgent.instructions,
                          originalLLM: workflow.defaultAgent.originalLLM,
                          tags: workflow.defaultAgent.tags.join(","),
                        })
                      }
                    >
                      Prefill agent form
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">Runs + observations</h2>
              <p className="text-sm text-muted-foreground">Recent Langfuse spans for the selected agent</p>
            </div>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-border bg-card p-4 space-y-2">
              <h3 className="text-base font-semibold">Latest runs</h3>
              <div className="space-y-3">
                {(data?.runs?.slice?.(0, 5) ?? []).map((run: any) => (
                  <article key={run.traceId} className="rounded-xl border border-border/50 bg-background px-3 py-2 text-sm">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="font-mono text-xs">{run.traceId}</span>
                      <span className="text-muted-foreground">agent: {run.agentId}</span>
                      <span className="text-muted-foreground">status: {run.status ?? "unknown"}</span>
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground mt-1">
                      <span>start: {formatDate(run.startedAt)}</span>
                      <span>end: {formatDate(run.completedAt)}</span>
                      {run.costUsd ? <span>cost: ${run.costUsd.toFixed(4)}</span> : null}
                    </div>
                  </article>
                ))}
              </div>
            </div>
            <div className="rounded-2xl border border-border bg-card p-4 overflow-x-auto">
              <h3 className="text-base font-semibold mb-2">Observations</h3>
              <table className="w-full min-w-[480px] text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase text-muted-foreground">
                    <th className="py-2 pr-4">Observation</th>
                    <th className="py-2 pr-4">Type</th>
                    <th className="py-2 pr-4">Status</th>
                    <th className="py-2 pr-4">Trace</th>
                    <th className="py-2 pr-4">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.observations?.slice?.(0, 8) ?? []).map((obs: any) => (
                    <tr key={obs.observationId} className="border-t border-border/60">
                      <td className="py-2 pr-4">{obs.name ?? "—"}</td>
                      <td className="py-2 pr-4">{obs.type ?? "—"}</td>
                      <td className="py-2 pr-4">{obs.status ?? "—"}</td>
                      <td className="py-2 pr-4 text-primary">
                        <a href={buildLangfuseUrl(obs.traceId)} target="_blank" rel="noreferrer" className="underline">
                          {obs.traceId}
                        </a>
                      </td>
                      <td className="py-2 pr-4">{formatDate(obs.startedAt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Agent graph</h2>
          {graphSample ? (
            <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-border bg-card p-5">
              {graphNodes.map((node, index) => (
                <div key={node.label} className="flex items-center gap-4">
                  <div className="rounded-xl border border-border/80 bg-background px-4 py-3 w-[240px]">
                    <p className="text-xs uppercase text-muted-foreground">{node.label}</p>
                    <p className="text-sm leading-relaxed mt-1 whitespace-pre-wrap">{node.content}</p>
                  </div>
                  {index < graphNodes.length - 1 && <span className="text-muted-foreground">→</span>}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No conversation captured yet.</p>
          )}
        </section>

        {data && (
          <section className="rounded-2xl border border-border bg-card p-4">
            <details>
              <summary className="cursor-pointer text-sm font-semibold">Raw JSON payload</summary>
              <pre className="mt-3 max-h-[320px] overflow-x-auto rounded-xl bg-background px-3 py-2 text-xs">
                {JSON.stringify(data, null, 2)}
              </pre>
            </details>
          </section>
        )}
      </div>
    </div>
  )
}
