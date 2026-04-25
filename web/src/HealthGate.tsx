import { useEffect, useState } from 'react';
import { api, type Health } from './api';
import './HealthGate.css';

type Props = { children: React.ReactNode };

export function HealthGate({ children }: Props) {
  const [health, setHealth] = useState<Health | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const h = await api.health.get();
        if (cancelled) return;
        setHealth(h);
        if (h.status === 'ready') return;
      } catch {
        // network blip during cold start; keep polling
      }
      if (!cancelled) timer = setTimeout(tick, 1500);
    };
    tick();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (health?.status === 'ready') return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [health?.status]);

  if (health?.status === 'ready') return <>{children}</>;

  const degraded = health?.status === 'degraded';
  return (
    <div className="health-gate">
      <div className="health-card">
        <div className={`health-spinner ${degraded ? 'degraded' : ''}`} />
        <h1>{degraded ? 'Server is degraded' : 'Waking up the server…'}</h1>
        <p className="health-detail">
          {degraded
            ? `Models failed to load: ${health?.error ?? 'unknown error'}`
            : `Cold start can take ~10–20s. ${elapsed}s elapsed.`}
        </p>
      </div>
    </div>
  );
}
