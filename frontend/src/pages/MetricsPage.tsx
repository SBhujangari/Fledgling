import { useQuery } from "@tanstack/react-query"
import { Navigation } from "@/components/navigation"
import { api } from "@/lib/api"
import type { ComparisonData, ComparisonTrack, MetricTrackStats } from "@/types"

export default function MetricsPage() {
  const { data, error, isLoading } = useQuery({
    queryKey: ["metrics-comparison"],
    queryFn: () => api.getMetricsComparison(),
  })

  const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`
  const formatParity = (val: number) => `${val.toFixed(1)}%`

  const renderMetricBlock = (label: string, stats?: MetricTrackStats | null) => {
    if (!stats) return null
    return (
      <div className="grid gap-3 rounded-2xl border border-border/70 bg-background p-4 text-sm">
        <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
        <div className="grid grid-cols-2 gap-4">
          <MetricStat title="Valid JSON" value={formatPercent(stats.json_valid_rate)} />
          <MetricStat title="Field F1" value={formatPercent(stats.field_f1)} />
          {typeof stats.field_precision === "number" && (
            <MetricStat title="Precision" value={formatPercent(stats.field_precision)} />
          )}
          {typeof stats.field_recall === "number" && (
            <MetricStat title="Recall" value={formatPercent(stats.field_recall)} />
          )}
        </div>
      </div>
    )
  }

  const renderParity = (track: ComparisonTrack) => {
    if (!track.delta) return null
    return (
      <div className="rounded-2xl border border-destructive/40 bg-destructive/5 p-4 text-sm">
        <h3 className="font-semibold text-destructive">Parity analysis</h3>
        <p className="text-2xl font-bold text-destructive">{formatParity(track.delta.f1_pct_of_azure)} of Azure performance</p>
        <p className="text-muted-foreground">
          Gap: {formatPercent(Math.abs(track.delta.f1_delta))} F1 difference
        </p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="w-full px-4 py-8 sm:px-6 lg:px-8">
        {isLoading && (
          <p className="text-sm text-muted-foreground">Loading metrics…</p>
        )}
        {error instanceof Error && (
          <p className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            Error: {error.message}
          </p>
        )}
        {data && (
          <>
            <header className="space-y-2 border-b border-border pb-6">
              <p className="text-sm text-muted-foreground">SLM vs LLM comparison</p>
              <h1 className="text-3xl font-semibold tracking-tight">Performance Headroom</h1>
              <p className="max-w-3xl text-sm text-muted-foreground">{data.summary.message}</p>
            </header>

            <section className="mt-8 grid gap-6 lg:grid-cols-2">
              <MetricCard title="Structured JSON Track" track={data.structured} renderMetricBlock={renderMetricBlock} renderParity={renderParity} />
              <MetricCard title="Tool Calling Track" track={data.toolcall} renderMetricBlock={renderMetricBlock} renderParity={renderParity} />
            </section>

            <section className="mt-8 rounded-3xl border border-border bg-amber-50 p-6 text-slate-900 dark:bg-amber-950 dark:text-amber-100">
              <h2 className="text-xl font-semibold">Recommendation</h2>
              <p className="mt-2 text-sm">{data.summary.recommendation}</p>
            </section>
          </>
        )}
      </main>
    </div>
  )
}

function MetricCard({
  title,
  track,
  renderMetricBlock,
  renderParity,
}: {
  title: string
  track: ComparisonTrack
  renderMetricBlock: (label: string, stats?: MetricTrackStats | null) => JSX.Element | null
  renderParity: (track: ComparisonTrack) => JSX.Element | null
}) {
  return (
    <article className="rounded-3xl border border-border bg-card p-6 shadow-sm">
      <h2 className="text-xl font-semibold">{title}</h2>
      <div className="mt-4 space-y-4">
        {renderMetricBlock("Azure LLM (baseline)", track.azure)}
        {renderMetricBlock("Fine-tuned SLM", track.slm)}
        {renderParity(track)}
      </div>
    </article>
  )
}

function MetricStat({ title, value }: { title: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{title}</p>
      <p className="text-xl font-semibold">{value}</p>
    </div>
  )
}
