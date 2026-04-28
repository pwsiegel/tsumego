import { useEffect, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { api, type TsumegoProblem } from './api';
import { ProblemEditor } from './ProblemEditor';

export function Review() {
  const { source: encSource = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  // ?status=rejected → review the user's previous rejections. Default is
  // unreviewed (the fresh review flow).
  const statusFilter = searchParams.get('status') || 'unreviewed';
  const [queue, setQueue] = useState<TsumegoProblem[] | null>(null);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Load the queue once: problems with the requested status in source order.
  useEffect(() => {
    (async () => {
      try {
        const problems = await api.tsumego.listProblems(source);
        setQueue(problems.filter((p) => p.status === statusFilter));
        setIndex(0);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, [source, statusFilter]);

  const backToCollection = () => navigate(`/collections/${encodeURIComponent(source)}/edit`);

  if (error) {
    return (
      <div style={{ maxWidth: '32rem', margin: '4rem auto', padding: '0 1.5rem' }}>
        <p style={{ color: '#c33' }}>Error: {error}</p>
        <p><a href="#" onClick={(e) => { e.preventDefault(); backToCollection(); }}>← back</a></p>
      </div>
    );
  }
  if (queue === null) {
    return (
      <div style={{ maxWidth: '32rem', margin: '4rem auto', padding: '0 1.5rem', color: '#666' }}>
        Loading…
      </div>
    );
  }

  if (queue.length === 0 || index >= queue.length) {
    const kind = statusFilter === 'rejected' ? 'rejected' : 'unreviewed';
    return (
      <div style={{ maxWidth: '32rem', margin: '4rem auto', padding: '0 1.5rem' }}>
        <h1>Review done</h1>
        <p style={{ color: '#666' }}>
          {queue.length === 0
            ? `Nothing ${kind} in this collection.`
            : `Reviewed ${queue.length} problem${queue.length === 1 ? '' : 's'}.`}
        </p>
        <p>
          <a href="#" onClick={(e) => { e.preventDefault(); backToCollection(); }}>
            ← back to collection
          </a>
        </p>
      </div>
    );
  }

  const current = queue[index];

  const goPrev = index > 0 ? () => setIndex(index - 1) : undefined;
  const goNext = index < queue.length - 1 ? () => setIndex(index + 1) : undefined;

  return (
    <ProblemEditor
      key={current.id}
      problem={current}
      onDecision={(newStatus) => {
        // Update the queue entry so if the user navigates back to this
        // problem, the editor reads the fresh status (and shows the ✓/✗
        // overlay) instead of the stale 'unreviewed' from initial load.
        setQueue((prev) => {
          if (!prev) return prev;
          const next = [...prev];
          next[index] = { ...next[index], status: newStatus };
          return next;
        });
        setIndex((i) => i + 1);
      }}
      onExit={backToCollection}
      onPrev={goPrev}
      onNext={goNext}
      label={`Problem ${index + 1} of ${queue.length} ${statusFilter}`}
    />
  );
}
