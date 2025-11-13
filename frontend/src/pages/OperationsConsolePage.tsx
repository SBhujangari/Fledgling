import { FormEvent, useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Navigation } from "@/components/navigation"
import { OpsTraceConsole } from "@/components/ops-trace-console"
import { api } from "@/lib/api"
import type { HfTokenMeta, SlmModelsResponse, UploadResult } from "@/types"

type ActiveTab = "ops" | "traces"

export default function OperationsConsolePage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("ops")
  const [repoId, setRepoId] = useState("")
  const [commitMessage, setCommitMessage] = useState("sync adapters via dashboard")
  const [branch, setBranch] = useState("")
  const [pathInRepo, setPathInRepo] = useState("")
  const [isPrivate, setIsPrivate] = useState(true)
  const [autoSubdir, setAutoSubdir] = useState(true)
  const [extraPaths, setExtraPaths] = useState("")
  const [hfTokenInput, setHfTokenInput] = useState("")

  const TARGETS = useMemo(
    () => [
      {
        label: "Structured adapter (slm_swap/04_ft/adapter_structured)",
        value: "slm_swap/04_ft/adapter_structured",
      },
      {
        label: "Tool-call adapter (slm_swap/04_ft/adapter_toolcall)",
        value: "slm_swap/04_ft/adapter_toolcall",
      },
    ],
    [],
  )

  const [selectedTargets, setSelectedTargets] = useState<string[]>(() => TARGETS.map((target) => target.value))
  const queryClient = useQueryClient()

  const {
    data: slmData,
    error: slmError,
    isFetching: isFetchingModels,
  } = useQuery<SlmModelsResponse>({
    queryKey: ["slm-models"],
    queryFn: () => api.getSlmCatalog(),
    refetchOnWindowFocus: false,
  })

  const {
    data: hfTokenMeta,
    error: hfTokenError,
  } = useQuery<HfTokenMeta>({
    queryKey: ["hf-token"],
    queryFn: () => api.getHfTokenMeta(),
    refetchOnWindowFocus: false,
  })

  const uploadMutation = useMutation({
    mutationFn: (payload: Parameters<typeof api.uploadHfArtifacts>[0]) => api.uploadHfArtifacts(payload),
  })

  const slmSelectionMutation = useMutation({
    mutationFn: (modelId: string) => api.selectSlmModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slm-models"] }).catch(() => {})
    },
  })

  const saveHfTokenMutation = useMutation({
    mutationFn: (token: string) => api.saveHfToken(token),
    onSuccess: () => {
      setHfTokenInput("")
      queryClient.invalidateQueries({ queryKey: ["hf-token"] }).catch(() => {})
    },
  })

  const clearHfTokenMutation = useMutation({
    mutationFn: () => api.clearHfToken(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["hf-token"] }).catch(() => {})
    },
  })

  const [pendingModelId, setPendingModelId] = useState("")
  useEffect(() => {
    if (slmData?.selectedModelId) {
      setPendingModelId(slmData.selectedModelId)
    }
  }, [slmData?.selectedModelId])

  const toggleTarget = (value: string) => {
    setSelectedTargets((current) => (current.includes(value) ? current.filter((item) => item !== value) : [...current, value]))
  }

  const handleUpload = (event: FormEvent) => {
    event.preventDefault()
    const supplementalPaths = extraPaths
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
    const payloadPaths = Array.from(new Set([...selectedTargets, ...supplementalPaths]))

    uploadMutation.mutate({
      repoId: repoId.trim(),
      commitMessage: commitMessage.trim() || undefined,
      branch: branch.trim() || undefined,
      pathInRepo: pathInRepo.trim() || undefined,
      private: isPrivate,
      autoSubdir,
      paths: payloadPaths,
    })
  }

  const uploadDisabled =
    uploadMutation.isPending || repoId.trim().length === 0 || (selectedTargets.length === 0 && extraPaths.trim().length === 0)
  const selectedSlm = slmData?.models.find((model) => model.id === slmData.selectedModelId)

  return (
    <div className="min-h-screen w-full bg-background text-foreground">
      <Navigation />
      <main className="w-full px-4 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-border pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Ops Console</p>
            <h1 className="text-3xl font-semibold tracking-tight">Agent Parity Playground</h1>
            <p className="text-sm text-muted-foreground">
              Manage SLM adapters, Hugging Face uploads, and Langfuse ingestion from a single dashboard.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setActiveTab("ops")}
              className={`rounded-full border px-4 py-2 text-sm font-medium transition ${
                activeTab === "ops" ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-muted"
              }`}
            >
              Ops Dashboard
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("traces")}
              className={`rounded-full border px-4 py-2 text-sm font-medium transition ${
                activeTab === "traces" ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-muted"
              }`}
            >
              Langfuse Trace Console
            </button>
          </div>
        </header>

        <div className="mt-8">
          {activeTab === "traces" ? (
            <OpsTraceConsole />
          ) : (
            <div className="grid gap-6 xl:grid-cols-2">
              <section className="rounded-3xl border border-border bg-card p-6 shadow-sm">
                <div className="space-y-1">
                  <h2 className="text-xl font-semibold">Hugging Face Token</h2>
                  <p className="text-sm text-muted-foreground">Store a single write token for automated uploads.</p>
                </div>
                <div className="mt-4 space-y-3">
                  <label className="space-y-1 text-sm font-medium">
                    Token (starts with <code>hf_</code>)
                    <input
                      type="password"
                      value={hfTokenInput}
                      onChange={(event) => setHfTokenInput(event.target.value)}
                      placeholder="hf_xxx..."
                      autoComplete="off"
                      className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                    />
                  </label>
                  <div className="flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => hfTokenInput.trim() && saveHfTokenMutation.mutate(hfTokenInput.trim())}
                      disabled={!hfTokenInput.trim() || saveHfTokenMutation.isPending}
                      className="rounded-xl border border-border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
                    >
                      {saveHfTokenMutation.isPending ? "Saving…" : "Save token"}
                    </button>
                    <button
                      type="button"
                      onClick={() => clearHfTokenMutation.mutate()}
                      disabled={clearHfTokenMutation.isPending || !hfTokenMeta?.hasToken}
                      className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/20 disabled:opacity-50"
                    >
                      {clearHfTokenMutation.isPending ? "Clearing…" : "Remove token"}
                    </button>
                  </div>
                  {hfTokenMeta && (
                    <p className="text-sm text-muted-foreground">
                      Status:{" "}
                      {hfTokenMeta.hasToken
                        ? `Stored (updated ${hfTokenMeta.updatedAt ? new Date(hfTokenMeta.updatedAt).toLocaleString() : "recently"})`
                        : "Missing"}
                    </p>
                  )}
                  {hfTokenError instanceof Error && (
                    <p className="text-sm text-destructive">{hfTokenError.message}</p>
                  )}
                  {saveHfTokenMutation.error instanceof Error && (
                    <p className="text-sm text-destructive">{saveHfTokenMutation.error.message}</p>
                  )}
                  {clearHfTokenMutation.error instanceof Error && (
                    <p className="text-sm text-destructive">{clearHfTokenMutation.error.message}</p>
                  )}
                </div>
              </section>

              <section className="rounded-3xl border border-border bg-card p-6 shadow-sm">
                <div className="space-y-1">
                  <h2 className="text-xl font-semibold">SLM Fine-Tune Selector</h2>
                  <p className="text-sm text-muted-foreground">
                    Choose the adapter checkpoint future training cycles should inherit.
                  </p>
                </div>

                {slmError instanceof Error && (
                  <p className="mt-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{slmError.message}</p>
                )}

                <label className="mt-4 block text-sm font-medium">
                  Available models
                  <select
                    value={pendingModelId}
                    onChange={(event) => setPendingModelId(event.target.value)}
                    disabled={isFetchingModels || !slmData}
                    className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                  >
                    <option value="" disabled>
                      {isFetchingModels ? "Loading models…" : "Select a model"}
                    </option>
                    {slmData?.models.map((model) => (
                      <option key={model.id} value={model.id} disabled={!model.available}>
                        {model.label}
                        {!model.available ? " (missing assets)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  disabled={!pendingModelId || slmSelectionMutation.isPending}
                  onClick={() => pendingModelId && slmSelectionMutation.mutate(pendingModelId)}
                  className="mt-4 inline-flex items-center justify-center rounded-xl border border-border px-4 py-2 text-sm font-semibold hover:bg-muted disabled:opacity-50"
                >
                  {slmSelectionMutation.isPending ? "Saving…" : "Set default SLM"}
                </button>
                {selectedSlm && (
                  <div className="mt-4 rounded-2xl border border-border/60 bg-background p-4 text-sm">
                    <p>
                      Current default: <strong>{selectedSlm.label}</strong>{" "}
                      <span className="text-muted-foreground">({selectedSlm.source})</span>
                    </p>
                    {slmData?.selectedAt && (
                      <p className="text-muted-foreground">
                        Updated: {new Date(slmData.selectedAt).toLocaleString()}
                      </p>
                    )}
                  </div>
                )}
                {slmSelectionMutation.error instanceof Error && (
                  <p className="mt-2 text-sm text-destructive">{slmSelectionMutation.error.message}</p>
                )}
                {slmData && (
                  <details className="mt-4 rounded-2xl border border-border/60 bg-background p-4 text-sm">
                    <summary className="cursor-pointer font-semibold">Model catalog</summary>
                    <ul className="mt-3 space-y-2">
                      {slmData.models.map((model) => (
                        <li key={model.id}>
                          <strong>{model.label}</strong> — {model.description ?? "No description"}
                          {!model.available && <span className="text-destructive"> (missing assets)</span>}
                          {model.capabilities && (
                            <span className="text-muted-foreground"> · Tracks: {model.capabilities.join(", ")}</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </section>

              <section className="rounded-3xl border border-border bg-card p-6 shadow-sm xl:col-span-2">
                <h2 className="text-xl font-semibold">Hugging Face Upload</h2>
                <p className="text-sm text-muted-foreground">
                  Commit adapters, eval assets, and supplemental files in one click.
                </p>

                <form onSubmit={handleUpload} className="mt-4 space-y-4">
                  <label className="block text-sm font-medium">
                    Repo ID (owner/name)
                    <input
                      type="text"
                      value={repoId}
                      onChange={(event) => setRepoId(event.target.value)}
                      placeholder="username/my-model"
                      required
                      className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                    />
                  </label>

                  <label className="block text-sm font-medium">
                    Commit message
                    <input
                      type="text"
                      value={commitMessage}
                      onChange={(event) => setCommitMessage(event.target.value)}
                      className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                    />
                  </label>

                  <div className="grid gap-4 md:grid-cols-3">
                    <label className="text-sm font-medium">
                      Branch (optional)
                      <input
                        type="text"
                        value={branch}
                        onChange={(event) => setBranch(event.target.value)}
                        placeholder="main"
                        className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                      />
                    </label>
                    <label className="text-sm font-medium md:col-span-2">
                      Path in repo (optional)
                      <input
                        type="text"
                        value={pathInRepo}
                        onChange={(event) => setPathInRepo(event.target.value)}
                        placeholder="adapters/latest"
                        className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
                      />
                    </label>
                  </div>

                  <div className="flex flex-wrap gap-6 text-sm">
                    <label className="flex items-center gap-2 font-medium">
                      <input type="checkbox" checked={isPrivate} onChange={() => setIsPrivate((prev) => !prev)} />
                      Private repo
                    </label>
                    <label className="flex items-center gap-2 font-medium">
                      <input type="checkbox" checked={autoSubdir} onChange={() => setAutoSubdir((prev) => !prev)} />
                      Auto sub-directory
                    </label>
                  </div>

                  <fieldset className="rounded-2xl border border-border/70 p-4">
                    <legend className="px-2 text-sm font-semibold">Tracked folders</legend>
                    <div className="mt-3 space-y-2 text-sm">
                      {TARGETS.map((target) => (
                        <label key={target.value} className="flex items-center gap-2">
                          <input type="checkbox" checked={selectedTargets.includes(target.value)} onChange={() => toggleTarget(target.value)} />
                          {target.label}
                        </label>
                      ))}
                    </div>
                  </fieldset>

                  <label className="block text-sm font-medium">
                    Extra paths (one per line, relative to repo root)
                    <textarea
                      value={extraPaths}
                      onChange={(event) => setExtraPaths(event.target.value)}
                      rows={3}
                      placeholder="slm_swap/05_eval/structured_slm_test.json"
                      className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 font-mono text-sm"
                    />
                  </label>

                  <button
                    type="submit"
                    disabled={uploadDisabled}
                    className="inline-flex items-center justify-center rounded-xl border border-border px-6 py-2 text-sm font-semibold hover:bg-muted disabled:opacity-50"
                  >
                    {uploadMutation.isPending ? "Uploading…" : "Upload to Hugging Face"}
                  </button>
                </form>

                {uploadMutation.error instanceof Error && (
                  <p className="mt-4 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                    {uploadMutation.error.message}
                  </p>
                )}

                {uploadMutation.data && (
                  <UploadSummaryCard upload={uploadMutation.data} />
                )}
              </section>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

function UploadSummaryCard({ upload }: { upload: UploadResult }) {
  return (
    <div className="mt-6 rounded-3xl border border-border/80 bg-background p-5 text-sm">
      <h3 className="text-lg font-semibold">Last upload</h3>
      <p className="mt-2 text-muted-foreground">
        Repo <strong>{upload.repoId}</strong> · {upload.private ? "Private" : "Public"} · {(upload.durationMs / 1000).toFixed(1)}s
      </p>
      <p className="text-xs text-muted-foreground">Command: {upload.command.join(" ")}</p>
      <details className="mt-3 rounded-2xl border border-border/60 bg-card p-3">
        <summary className="cursor-pointer font-semibold">stdout</summary>
        <pre className="mt-2 max-h-60 overflow-auto rounded-xl bg-muted/40 p-3 text-xs">{upload.stdout || "(empty)"}</pre>
      </details>
      {upload.stderr && (
        <details className="mt-3 rounded-2xl border border-border/60 bg-card p-3">
          <summary className="cursor-pointer font-semibold">stderr</summary>
          <pre className="mt-2 max-h-60 overflow-auto rounded-xl bg-muted/40 p-3 text-xs text-destructive">{upload.stderr}</pre>
        </details>
      )}
    </div>
  )
}
