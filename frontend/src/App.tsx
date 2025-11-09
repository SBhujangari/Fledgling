import { FormEvent, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

interface TracesResponse {
  runs: unknown[];
  observations: unknown[];
  generations: unknown[];
}

interface UploadResult {
  repoId: string;
  repoType: string;
  private: boolean;
  paths: string[];
  command: string[];
  stdout: string;
  stderr: string;
  durationMs: number;
}

interface UploadRequestPayload {
  repoId: string;
  paths: string[];
  commitMessage?: string;
  branch?: string;
  pathInRepo?: string;
  private?: boolean;
  autoSubdir?: boolean;
}

interface SlmModel {
  id: string;
  label: string;
  source: string;
  description?: string;
  capabilities?: string[];
  available: boolean;
  location: string | null;
}

interface SlmModelsResponse {
  models: SlmModel[];
  selectedModelId: string | null;
  selectedAt: string | null;
}

interface HfTokenMeta {
  hasToken: boolean;
  updatedAt: string | null;
}

async function fetchTraces(): Promise<TracesResponse> {
  const response = await fetch('/api/traces');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to fetch traces');
  }
  return response.json();
}

async function fetchSlmModels(): Promise<SlmModelsResponse> {
  const response = await fetch('/api/slm/models');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to load SLM catalog');
  }
  return response.json();
}

async function fetchHfTokenMeta(): Promise<HfTokenMeta> {
  const response = await fetch('/api/hf/token');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to load HF token status');
  }
  return response.json();
}

export default function App() {
  const [lastFetchedAt, setLastFetchedAt] = useState<string | null>(null);
  const [repoId, setRepoId] = useState('');
  const [commitMessage, setCommitMessage] = useState('sync adapters via dashboard');
  const [branch, setBranch] = useState('');
  const [pathInRepo, setPathInRepo] = useState('');
  const [isPrivate, setIsPrivate] = useState(true);
  const [autoSubdir, setAutoSubdir] = useState(true);
  const TARGETS = useMemo(
    () => [
      {
        label: 'Structured adapter (slm_swap/04_ft/adapter_structured)',
        value: 'slm_swap/04_ft/adapter_structured',
      },
      {
        label: 'Tool-call adapter (slm_swap/04_ft/adapter_toolcall)',
        value: 'slm_swap/04_ft/adapter_toolcall',
      },
    ],
    [],
  );
  const [selectedTargets, setSelectedTargets] = useState<string[]>(TARGETS.map((target) => target.value));
  const [extraPaths, setExtraPaths] = useState('');
  const queryClient = useQueryClient();
  const { data, error, isFetching, refetch } = useQuery({
    queryKey: ['traces'],
    queryFn: fetchTraces,
    enabled: false,
    retry: false,
  });
  const {
    data: slmData,
    error: slmError,
    isFetching: isFetchingModels,
  } = useQuery({
    queryKey: ['slm-models'],
    queryFn: fetchSlmModels,
    refetchOnWindowFocus: false,
  });
  const {
    data: hfTokenMeta,
    error: hfTokenError,
  } = useQuery({
    queryKey: ['hf-token'],
    queryFn: fetchHfTokenMeta,
    refetchOnWindowFocus: false,
  });
  const uploadMutation = useMutation<UploadResult, Error, UploadRequestPayload>({
    mutationFn: async (payload: UploadRequestPayload) => {
      const response = await fetch('/api/hf/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = (await response.json()) as { ok: boolean; result?: UploadResult; error?: string };
      if (!response.ok || !data.ok || !data.result) {
        throw new Error(data.error || 'Failed to upload artifacts');
      }
      return data.result;
    },
  });
  const slmSelectionMutation = useMutation(
    async (modelId: string) => {
      const response = await fetch('/api/slm/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId }),
      });
      const data = (await response.json()) as { ok: boolean; selection?: { modelId: string; selectedAt: string }; error?: string };
      if (!response.ok || !data.ok || !data.selection) {
        throw new Error(data.error || 'Failed to set model selection');
      }
      return data.selection;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['slm-models'] }).catch(() => {});
      },
    },
  );
  const [pendingModelId, setPendingModelId] = useState('');
  useEffect(() => {
    if (slmData?.selectedModelId) {
      setPendingModelId(slmData.selectedModelId);
    }
  }, [slmData?.selectedModelId]);
  const [hfTokenInput, setHfTokenInput] = useState('');
  const saveHfTokenMutation = useMutation(
    async (token: string) => {
      const response = await fetch('/api/hf/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      const data = (await response.json()) as { ok: boolean; updatedAt?: string; error?: string };
      if (!response.ok || !data.ok) {
        throw new Error(data.error || 'Failed to save token');
      }
      return data.updatedAt;
    },
    {
      onSuccess: () => {
        setHfTokenInput('');
        queryClient.invalidateQueries({ queryKey: ['hf-token'] }).catch(() => {});
      },
    },
  );
  const clearHfTokenMutation = useMutation(
    async () => {
      const response = await fetch('/api/hf/token', { method: 'DELETE' });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to clear token');
      }
      return true;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['hf-token'] }).catch(() => {});
      },
    },
  );

  const handleFetch = async () => {
    const result = await refetch();
    if (result.data) {
      setLastFetchedAt(new Date().toLocaleString());
    }
  };

  const toggleTarget = (value: string) => {
    setSelectedTargets((current) => (current.includes(value) ? current.filter((item) => item !== value) : [...current, value]));
  };

  const handleUpload = (event: FormEvent) => {
    event.preventDefault();
    const supplementalPaths = extraPaths
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    const payloadPaths = Array.from(new Set([...selectedTargets, ...supplementalPaths]));

    uploadMutation.mutate({
      repoId: repoId.trim(),
      commitMessage: commitMessage.trim() || undefined,
      branch: branch.trim() || undefined,
      pathInRepo: pathInRepo.trim() || undefined,
      private: isPrivate,
      autoSubdir,
      paths: payloadPaths,
    });
  };

  const uploadDisabled = uploadMutation.isPending || repoId.trim().length === 0 || (selectedTargets.length === 0 && extraPaths.trim().length === 0);
  const selectedSlm = slmData?.models.find((model) => model.id === slmData.selectedModelId);

  return (
    <main style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: '2rem', maxWidth: 1100, margin: '0 auto' }}>
      <h1>Operations Console</h1>
      <p>Trigger trace ingestion or push fresh adapters/eval assets to your private Hugging Face repo.</p>
      <div style={{ display: 'grid', gap: '2rem', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', marginTop: '1.5rem' }}>
        <section style={{ border: '1px solid #e2e8f0', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h2>Langfuse Trace Fetch</h2>
          <p>Press the button to request the latest completed traces for the configured agent.</p>
          <button type="button" onClick={handleFetch} disabled={isFetching} style={{ padding: '0.75rem 1.5rem' }}>
            {isFetching ? 'Loading…' : 'Fetch traces'}
          </button>

          {lastFetchedAt && <p style={{ marginTop: '1rem' }}>Last fetched: {lastFetchedAt}</p>}

          {error instanceof Error && (
            <pre style={{ color: 'red', marginTop: '1rem' }}>{error.message}</pre>
          )}

          {data && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3>Response</h3>
              <pre style={{ background: '#0f172a', color: '#f8fafc', padding: '1rem', borderRadius: '0.5rem', overflowX: 'auto', maxHeight: 320 }}>
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </section>

        <section style={{ border: '1px solid #e2e8f0', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h2>Hugging Face Token</h2>
          <p>Store your write-scoped HF token once; backend routes will reuse it for uploads.</p>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>
            Token (starts with <code>hf_</code>)
            <input
              type="password"
              value={hfTokenInput}
              onChange={(event) => setHfTokenInput(event.target.value)}
              placeholder="hf_xxx..."
              autoComplete="off"
              style={{ width: '100%', marginTop: '0.35rem', padding: '0.5rem' }}
            />
          </label>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              type="button"
              onClick={() => saveHfTokenMutation.mutate(hfTokenInput.trim())}
              disabled={!hfTokenInput.trim() || saveHfTokenMutation.isPending}
              style={{ padding: '0.6rem 1.25rem' }}
            >
              {saveHfTokenMutation.isPending ? 'Saving…' : 'Save token'}
            </button>
            <button
              type="button"
              onClick={() => clearHfTokenMutation.mutate()}
              disabled={clearHfTokenMutation.isPending || !hfTokenMeta?.hasToken}
              style={{ padding: '0.6rem 1.25rem', background: '#fee2e2', border: '1px solid #fecaca' }}
            >
              {clearHfTokenMutation.isPending ? 'Clearing…' : 'Remove token'}
            </button>
          </div>
          {hfTokenMeta && (
            <p style={{ marginTop: '0.75rem', color: '#475569' }}>
              Status: {hfTokenMeta.hasToken ? `Stored (updated ${hfTokenMeta.updatedAt ? new Date(hfTokenMeta.updatedAt).toLocaleString() : 'recently'})` : 'Missing'}
            </p>
          )}
          {hfTokenError instanceof Error && (
            <pre style={{ color: 'red', marginTop: '0.75rem' }}>{hfTokenError.message}</pre>
          )}
          {saveHfTokenMutation.error instanceof Error && (
            <pre style={{ color: 'red', marginTop: '0.75rem' }}>{saveHfTokenMutation.error.message}</pre>
          )}
          {clearHfTokenMutation.error instanceof Error && (
            <pre style={{ color: 'red', marginTop: '0.75rem' }}>{clearHfTokenMutation.error.message}</pre>
          )}
        </section>

        <section style={{ border: '1px solid #e2e8f0', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h2>SLM Fine-Tune Selector</h2>
          <p>Choose which base/fine-tuned checkpoint the next training cycle should start from.</p>
          {slmError instanceof Error && (
            <pre style={{ color: 'red', marginBottom: '0.75rem' }}>{slmError.message}</pre>
          )}
          <label style={{ display: 'block', marginBottom: '0.75rem' }}>
            Available models
            <select
              value={pendingModelId}
              onChange={(event) => setPendingModelId(event.target.value)}
              disabled={isFetchingModels || !slmData}
              style={{ width: '100%', marginTop: '0.35rem', padding: '0.5rem' }}
            >
              <option value="" disabled>
                {isFetchingModels ? 'Loading models…' : 'Select a model'}
              </option>
              {slmData?.models.map((model) => (
                <option key={model.id} value={model.id} disabled={!model.available}>
                  {model.label}
                  {!model.available ? ' (missing assets)' : ''}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={!pendingModelId || slmSelectionMutation.isPending}
            onClick={() => slmSelectionMutation.mutate(pendingModelId)}
            style={{ padding: '0.6rem 1.25rem' }}
          >
            {slmSelectionMutation.isPending ? 'Saving…' : 'Set default SLM'}
          </button>
          {selectedSlm && (
            <div style={{ marginTop: '1rem', fontSize: '0.95rem' }}>
              <p>
                Current default: <strong>{selectedSlm.label}</strong>{' '}
                <span style={{ color: '#475569' }}>({selectedSlm.source})</span>
              </p>
              {slmData?.selectedAt && <p style={{ color: '#475569' }}>Updated: {new Date(slmData.selectedAt).toLocaleString()}</p>}
            </div>
          )}
          {slmSelectionMutation.error instanceof Error && (
            <pre style={{ color: 'red', marginTop: '0.75rem' }}>{slmSelectionMutation.error.message}</pre>
          )}
          {slmData && (
            <details style={{ marginTop: '1rem' }}>
              <summary>Model catalog</summary>
              <ul style={{ marginTop: '0.5rem', paddingLeft: '1.25rem' }}>
                {slmData.models.map((model) => (
                  <li key={model.id} style={{ marginBottom: '0.5rem' }}>
                    <strong>{model.label}</strong> — {model.description}{' '}
                    {!model.available && <span style={{ color: '#dc2626' }}>(missing assets)</span>}
                    {model.capabilities && (
                      <span style={{ color: '#475569' }}> · Tracks: {model.capabilities.join(', ')}</span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>

        <section style={{ border: '1px solid #e2e8f0', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h2>Hugging Face Upload</h2>
          <form onSubmit={handleUpload}>
            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              Repo ID (owner/name)
              <input
                type="text"
                value={repoId}
                onChange={(event) => setRepoId(event.target.value)}
                placeholder="username/my-model"
                style={{ width: '100%', marginTop: '0.25rem', padding: '0.5rem' }}
                required
              />
            </label>

            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              Commit message
              <input
                type="text"
                value={commitMessage}
                onChange={(event) => setCommitMessage(event.target.value)}
                placeholder="sync adapters via dashboard"
                style={{ width: '100%', marginTop: '0.25rem', padding: '0.5rem' }}
              />
            </label>

            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              Branch (optional)
              <input
                type="text"
                value={branch}
                onChange={(event) => setBranch(event.target.value)}
                placeholder="main"
                style={{ width: '100%', marginTop: '0.25rem', padding: '0.5rem' }}
              />
            </label>

            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              Path in repo (optional)
              <input
                type="text"
                value={pathInRepo}
                onChange={(event) => setPathInRepo(event.target.value)}
                placeholder="adapters/latest"
                style={{ width: '100%', marginTop: '0.25rem', padding: '0.5rem' }}
              />
            </label>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem' }}>
              <label style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                <input type="checkbox" checked={isPrivate} onChange={() => setIsPrivate((prev) => !prev)} />
                Private repo
              </label>
              <label style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                <input type="checkbox" checked={autoSubdir} onChange={() => setAutoSubdir((prev) => !prev)} />
                Auto sub-directory
              </label>
            </div>

            <fieldset style={{ border: '1px solid #cbd5f5', padding: '0.75rem', borderRadius: '0.5rem', marginBottom: '0.75rem' }}>
              <legend style={{ padding: '0 0.5rem' }}>Tracked folders</legend>
              {TARGETS.map((target) => (
                <label key={target.value} style={{ display: 'flex', gap: '0.35rem', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <input type="checkbox" checked={selectedTargets.includes(target.value)} onChange={() => toggleTarget(target.value)} />
                  {target.label}
                </label>
              ))}
            </fieldset>

            <label style={{ display: 'block', marginBottom: '0.75rem' }}>
              Extra paths (one per line, relative to repo root)
              <textarea
                value={extraPaths}
                onChange={(event) => setExtraPaths(event.target.value)}
                rows={3}
                style={{ width: '100%', marginTop: '0.25rem', padding: '0.5rem', fontFamily: 'monospace' }}
                placeholder="slm_swap/05_eval/structured_slm_test.json"
              />
            </label>

            <button type="submit" disabled={uploadDisabled} style={{ padding: '0.75rem 1.5rem' }}>
              {uploadMutation.isPending ? 'Uploading…' : 'Upload to Hugging Face'}
            </button>
          </form>

          {uploadMutation.error instanceof Error && (
            <pre style={{ color: 'red', marginTop: '1rem' }}>{uploadMutation.error.message}</pre>
          )}

          {uploadMutation.data && (
            <div style={{ marginTop: '1rem' }}>
              <h3>Last upload</h3>
              <p>
                Repo <strong>{uploadMutation.data.repoId}</strong> ·{' '}
                {uploadMutation.data.private ? 'Private' : 'Public'} ·{' '}
                {(uploadMutation.data.durationMs / 1000).toFixed(1)}s
              </p>
              <p style={{ fontSize: '0.9rem', color: '#475569' }}>Command: {uploadMutation.data.command.join(' ')}</p>
              <details style={{ marginTop: '0.5rem' }}>
                <summary>stdout</summary>
                <pre style={{ background: '#0f172a', color: '#f8fafc', padding: '0.75rem', borderRadius: '0.5rem', overflowX: 'auto' }}>
                  {uploadMutation.data.stdout || '(empty)'}
                </pre>
              </details>
              {uploadMutation.data.stderr && (
                <details style={{ marginTop: '0.5rem' }}>
                  <summary>stderr</summary>
                  <pre style={{ background: '#1f2937', color: '#f8fafc', padding: '0.75rem', borderRadius: '0.5rem', overflowX: 'auto' }}>
                    {uploadMutation.data.stderr}
                  </pre>
                </details>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
