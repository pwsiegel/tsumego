import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api, type TsumegoProblem } from './api';
import { ProblemEditor } from './ProblemEditor';

/** Single-problem editor, reached by clicking a tile on the collection
 * page. After a decision, navigate back to the collection list. */
export function ProblemDetail() {
  const { source: encSource = '', id = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [problem, setProblem] = useState<TsumegoProblem | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tsumego.getProblem(id)
      .then(setProblem)
      .catch((e) => setError(String(e)));
  }, [id]);

  const backToCollection = () => navigate(`/collections/${encodeURIComponent(source)}/edit`);

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
      onDecision={() => backToCollection()}
      onExit={backToCollection}
      label={`Problem ${problem.source_board_idx + 1}`}
    />
  );
}
