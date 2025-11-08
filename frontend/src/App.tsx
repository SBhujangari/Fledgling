import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

interface TracesResponse {
  runs: unknown[];
  observations: unknown[];
  generations: unknown[];
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
  const { data, error, isFetching, refetch } = useQuery({
    queryKey: ['traces'],
    queryFn: fetchTraces,
    enabled: false,
    retry: false,
  });

  const handleFetch = async () => {
    const result = await refetch();
    if (result.data) {
      setLastFetchedAt(new Date().toLocaleString());
    }
  };

  return (
    <main style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: '2rem', maxWidth: 900, margin: '0 auto' }}>
      <h1>Tracer Demo Console</h1>
      <p>Press the button to request the latest completed traces for the configured agent.</p>
      <button type="button" onClick={handleFetch} disabled={isFetching} style={{ padding: '0.75rem 1.5rem' }}>
        {isFetching ? 'Loading…' : 'Fetch traces'}
      </button>

      {lastFetchedAt && <p style={{ marginTop: '1rem' }}>Last fetched: {lastFetchedAt}</p>}

      {error instanceof Error && (
        <pre style={{ color: 'red', marginTop: '1rem' }}>{error.message}</pre>
      )}

      {data && (
        <section style={{ marginTop: '1.5rem' }}>
          <h2>Response</h2>
          <pre style={{ background: '#0f172a', color: '#f8fafc', padding: '1rem', borderRadius: '0.5rem', overflowX: 'auto' }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </section>
      )}
    </main>
  );
}
