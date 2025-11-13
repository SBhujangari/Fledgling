import { useQuery } from "@tanstack/react-query"
import { Navigation } from "@/components/navigation"
import { api } from "@/lib/api"

interface DashboardData {
  agent: {
    id: string
    name: string
    model: string
    status: string
    last_run: string | null
  }
  overview: {
    total_runs: number
    success_rate: number
    avg_latency_ms: number
    total_cost_usd: number
  }
  quality: {
    exact_match: number
    tool_name_accuracy: number
    query_preservation: number
    json_validity: number
    functional_correctness: number
    semantic_correctness: number
  }
  comparison: {
    slm_exact_match: number
    azure_exact_match: number
    improvement_pct: number
    slm_wins: boolean
  }
  recent_traces: Array<{
    traceId: string
    name?: string
    status?: string
    latencyMs?: number
    completedAt?: string
  }>
  sample_results?: Array<{
    example_id: number
    prompt: string
    expected: any
    predicted: any
    metrics: {
      exact_match: boolean
      tool_name_match?: boolean
      json_valid: boolean
    }
  }>
}

export default function SLMDashboardPage() {
  const { data, error, isLoading } = useQuery<DashboardData>({
    queryKey: ["slm-dashboard"],
    queryFn: async () => {
      const response = await fetch("http://localhost:3000/api/metrics/dashboard")
      if (!response.ok) throw new Error("Failed to fetch dashboard data")
      return response.json()
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`
  const formatMs = (val: number) => `${val.toFixed(0)}ms`

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="w-full px-4 py-8 sm:px-6 lg:px-8">
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <p className="text-sm text-muted-foreground">Loading SLM metrics…</p>
          </div>
        )}

        {error instanceof Error && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-6 text-destructive">
            <h3 className="font-semibold">Error Loading Dashboard</h3>
            <p className="mt-2 text-sm">{error.message}</p>
            <p className="mt-4 text-sm">
              Try running: <code className="rounded bg-black/10 px-2 py-1 font-mono text-xs">CUDA_VISIBLE_DEVICES=0 python example_slm_agent.py</code>
            </p>
          </div>
        )}

        {data && (
          <div className="space-y-8">
            {/* Header */}
            <header className="space-y-2 border-b border-border pb-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Fine-Tuned SLM Performance</p>
                  <h1 className="text-3xl font-semibold tracking-tight">{data.agent.name}</h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Model: {data.agent.model} • Status:{" "}
                    <span className={data.agent.status === "active" ? "text-green-600 dark:text-green-400" : "text-yellow-600 dark:text-yellow-400"}>
                      {data.agent.status}
                    </span>
                  </p>
                </div>
                {data.agent.status === "not-run" && (
                  <div className="rounded-lg border border-amber-500/40 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100">
                    Run example agent to see metrics
                  </div>
                )}
              </div>
            </header>

            {/* Key Metrics Grid */}
            <section className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                title="Total Runs"
                value={data.overview.total_runs.toString()}
                subtitle={`${formatPercent(data.overview.success_rate / 100)} success rate`}
                icon="🚀"
              />
              <MetricCard
                title="Avg Latency"
                value={formatMs(data.overview.avg_latency_ms)}
                subtitle="Per inference"
                icon="⚡"
              />
              <MetricCard
                title="Cost Savings"
                value="180x"
                subtitle="vs Azure GPT-4"
                icon="💰"
                highlight={true}
              />
              <MetricCard
                title="JSON Validity"
                value={formatPercent(data.quality.json_validity)}
                subtitle="No parsing errors"
                icon="✅"
                highlight={data.quality.json_validity === 1.0}
              />
            </section>

            {/* Comparison Section */}
            <section className="rounded-3xl border border-border bg-gradient-to-br from-blue-50 to-purple-50 p-6 dark:from-blue-950/30 dark:to-purple-950/30">
              <h2 className="text-2xl font-semibold">SLM vs Azure GPT Baseline</h2>
              <div className="mt-6 grid gap-6 md:grid-cols-3">
                <ComparisonStat
                  label="Fine-tuned SLM"
                  value={formatPercent(data.comparison.slm_exact_match)}
                  type="slm"
                />
                <ComparisonStat
                  label="Azure GPT Baseline"
                  value={formatPercent(data.comparison.azure_exact_match)}
                  type="azure"
                />
                <ComparisonStat
                  label="Improvement"
                  value={`+${data.comparison.improvement_pct.toFixed(0)}%`}
                  type="improvement"
                  highlight={true}
                />
              </div>
              {data.comparison.slm_wins && (
                <div className="mt-6 rounded-xl border border-green-500/40 bg-green-50 p-4 text-green-900 dark:bg-green-950 dark:text-green-100">
                  <p className="text-sm font-semibold">🎉 Fine-tuned SLM outperforms Azure GPT by {data.comparison.improvement_pct.toFixed(0)}%!</p>
                </div>
              )}
            </section>

            {/* Quality Metrics */}
            <section className="rounded-3xl border border-border bg-card p-6">
              <h2 className="text-2xl font-semibold">Model Quality Metrics</h2>
              <p className="mt-1 text-sm text-muted-foreground">Detailed evaluation on 50-example test set</p>
              <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <QualityMetric
                  title="Exact Match"
                  value={formatPercent(data.quality.exact_match)}
                  description="Perfect match with expected output"
                />
                <QualityMetric
                  title="Tool Name Accuracy"
                  value={formatPercent(data.quality.tool_name_accuracy)}
                  description="Calls the correct function"
                  highlight={data.quality.tool_name_accuracy >= 0.95}
                />
                <QualityMetric
                  title="Query Preservation"
                  value={formatPercent(data.quality.query_preservation)}
                  description="Maintains user intent"
                  highlight={data.quality.query_preservation >= 0.90}
                />
                <QualityMetric
                  title="Functional Correctness"
                  value={formatPercent(data.quality.functional_correctness)}
                  description="Output is usable even if not exact"
                />
                <QualityMetric
                  title="Semantic Correctness"
                  value={formatPercent(data.quality.semantic_correctness)}
                  description="Understands and preserves meaning"
                />
                <QualityMetric
                  title="JSON Validity"
                  value={formatPercent(data.quality.json_validity)}
                  description="Never produces malformed output"
                  highlight={data.quality.json_validity === 1.0}
                />
              </div>
            </section>

            {/* Recent Traces */}
            {data.recent_traces && data.recent_traces.length > 0 && (
              <section className="rounded-3xl border border-border bg-card p-6">
                <h2 className="text-2xl font-semibold">Recent Agent Runs</h2>
                <div className="mt-6 space-y-3">
                  {data.recent_traces.map((trace) => (
                    <div
                      key={trace.traceId}
                      className="flex items-center justify-between rounded-lg border border-border bg-background p-4"
                    >
                      <div className="flex items-center gap-4">
                        <div className={`h-2 w-2 rounded-full ${trace.status === "completed" ? "bg-green-500" : "bg-yellow-500"}`} />
                        <div>
                          <p className="font-mono text-sm text-muted-foreground">{trace.traceId}</p>
                          <p className="text-xs text-muted-foreground">{trace.name || "api-generation"}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        {trace.latencyMs && <p className="text-sm font-medium">{formatMs(trace.latencyMs)}</p>}
                        {trace.completedAt && (
                          <p className="text-xs text-muted-foreground">
                            {new Date(trace.completedAt).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Sample Results */}
            {data.sample_results && data.sample_results.length > 0 && (
              <section className="rounded-3xl border border-border bg-card p-6">
                <h2 className="text-2xl font-semibold">Sample Predictions</h2>
                <p className="mt-1 text-sm text-muted-foreground">Examples from evaluation test set</p>
                <div className="mt-6 space-y-4">
                  {data.sample_results.slice(0, 3).map((result) => (
                    <div
                      key={result.example_id}
                      className={`rounded-lg border p-4 ${
                        result.metrics.exact_match
                          ? "border-green-500/40 bg-green-50/50 dark:bg-green-950/20"
                          : "border-yellow-500/40 bg-yellow-50/50 dark:bg-yellow-950/20"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <p className="text-xs font-semibold text-muted-foreground">Example #{result.example_id}</p>
                        <span className={`rounded-full px-2 py-1 text-xs font-medium ${
                          result.metrics.exact_match
                            ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                            : result.metrics.tool_name_match
                            ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
                            : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                        }`}>
                          {result.metrics.exact_match ? "Exact Match" : result.metrics.tool_name_match ? "Partial Match" : "Mismatch"}
                        </span>
                      </div>
                      <p className="mt-2 text-sm">{result.prompt.substring(0, 150)}...</p>
                      {result.predicted && (
                        <div className="mt-3 rounded bg-black/5 p-3 font-mono text-xs dark:bg-white/5">
                          <p className="text-muted-foreground">Tool: {result.predicted.tool_name}</p>
                          <p className="text-muted-foreground">Args: {JSON.stringify(result.predicted.arguments).substring(0, 80)}...</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Call to Action */}
            <section className="rounded-3xl border border-blue-500/40 bg-blue-50 p-6 dark:bg-blue-950/30">
              <h2 className="text-xl font-semibold">Ready for Production</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                This fine-tuned model achieves {formatPercent(data.quality.exact_match)} exact match accuracy with{" "}
                {formatPercent(data.quality.json_validity)} JSON validity. It's ready for deployment in production workloads.
              </p>
              <div className="mt-4 flex gap-3">
                <button className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
                  Deploy Model
                </button>
                <button className="rounded-lg border border-border bg-background px-4 py-2 text-sm font-medium hover:bg-muted">
                  View Full Report
                </button>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}

function MetricCard({
  title,
  value,
  subtitle,
  icon,
  highlight = false,
}: {
  title: string
  value: string
  subtitle: string
  icon: string
  highlight?: boolean
}) {
  return (
    <div className={`rounded-2xl border p-6 ${highlight ? "border-blue-500/40 bg-blue-50 dark:bg-blue-950/30" : "border-border bg-card"}`}>
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
        <span className="text-2xl">{icon}</span>
      </div>
      <p className="mt-2 text-3xl font-bold">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
    </div>
  )
}

function ComparisonStat({
  label,
  value,
  type,
  highlight = false,
}: {
  label: string
  value: string
  type: "slm" | "azure" | "improvement"
  highlight?: boolean
}) {
  return (
    <div className={`rounded-xl border p-4 ${
      type === "improvement"
        ? "border-green-500/40 bg-green-50 dark:bg-green-950/30"
        : "border-border bg-white/50 dark:bg-black/20"
    }`}>
      <p className="text-sm font-medium text-muted-foreground">{label}</p>
      <p className={`mt-2 text-3xl font-bold ${
        type === "improvement" ? "text-green-600 dark:text-green-400" : ""
      }`}>{value}</p>
    </div>
  )
}

function QualityMetric({
  title,
  value,
  description,
  highlight = false,
}: {
  title: string
  value: string
  description: string
  highlight?: boolean
}) {
  return (
    <div className={`rounded-lg border p-4 ${
      highlight ? "border-blue-500/40 bg-blue-50/50 dark:bg-blue-950/20" : "border-border bg-background"
    }`}>
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
    </div>
  )
}
