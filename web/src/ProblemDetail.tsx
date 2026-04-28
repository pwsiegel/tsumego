import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api, type TsumegoProblem } from './api';
import { ProblemEditor } from './ProblemEditor';

/** Single-problem editor, reached by clicking a tile on the collection
 * page. After a decision, advance to the next problem in source order;
 * fall back to the collection when there is no next. */
export function ProblemDetail() {
  const { source: encSource = '', id = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [problem, setProblem] = useState<TsumegoProblem | null>(null);
  const [siblings, setSiblings] = useState<TsumegoProblem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tsumego.getProblem(id)
      .then(setProblem)
      .catch((e) => setError(String(e)));
  }, [id]);

  useEffect(() => {
    api.tsumego.listProblems(source)
      .then(setSiblings)
      .catch((e) => setError(String(e)));
  }, [source]);

  const backToCollection = () => navigate(`/collections/${encodeURIComponent(source)}/edit`);
  const goToProblem = (pid: string) =>
    navigate(`/collections/${encodeURIComponent(source)}/problem/${pid}`);

  const index = siblings ? siblings.findIndex((p) => p.id === id) : -1;
  const prev = siblings && index > 0 ? siblings[index - 1] : null;
  const next = siblings && index >= 0 && index < siblings.length - 1
    ? siblings[index + 1] : null;

  if (error) {
    return (
      <div style={{ maxWidth: '32rem', margin: '4rem auto', padding: '0 1.5rem' }}>
        <p style={{ color: '#c33' }}>Error: {error}</p>
        <p><a href="#" onClick={(e) => { e.preventDefault(); backToCollection(); }}>← back</a></p>
      </div>
    );
  }
  if (!problem) {
    return <div style={{ maxWidth: '32rem', margin: '4rem auto', padding: '0 1.5rem', color: '#666' }}>Loading…</div>;
  }

  return (
    <ProblemEditor
      key={problem.id}
      problem={problem}
      onDecision={() => (next ? goToProblem(next.id) : backToCollection())}
      onExit={backToCollection}
      onPrev={prev ? () => goToProblem(prev.id) : undefined}
      onNext={next ? () => goToProblem(next.id) : undefined}
      label={`Problem ${problem.source_board_idx + 1}`}
    />
  );
}
