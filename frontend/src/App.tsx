import { FormEvent, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';

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

async function fetchTraces(): Promise<TracesResponse> {
  const response = await fetch('/api/traces');
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to fetch traces');
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
  const { data, error, isFetching, refetch } = useQuery({
    queryKey: ['traces'],
    queryFn: fetchTraces,
    enabled: false,
    retry: false,
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
