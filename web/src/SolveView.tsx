import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Board } from './Board';
import { api, type Attempt, type LinkedUser, type ProblemStatus, type TsumegoProblem } from './api';
import { computeNumberedOverlay, type MovePoint } from './numberedMoves';
import type { Stone } from './types';
import './SolveView.css';

/** Tap-to-place numbered-move solver. Numbers paint on top of the
 * problem's initial position; no stones get added by the student. The
 * full sequence is sent to the teacher's review queue on submit. */
export function SolveView() {
  const { source: encSource = '', id = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const fromSubmission = searchParams.get('from_submission');

  const [problem, setProblem] = useState<TsumegoProblem | null>(null);
  const [attempts, setAttempts] = useState<Attempt[] | null>(null);
  const [teachers, setTeachers] = useState<LinkedUser[]>([]);
  const [siblings, setSiblings] = useState<TsumegoProblem[] | null>(null);
  const [moves, setMoves] = useState<MovePoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setProblem(null);
    setAttempts(null);
    setMoves([]);
    setFlash(null);
    setError(null);
    Promise.all([
      api.tsumego.getProblem(id),
      api.study.listAttempts(id),
      api.study.listTeachers().catch(() => [] as LinkedUser[]),
    ])
      .then(([p, a, ts]) => {
        if (cancelled) return;
        setProblem(p);
        setAttempts(a);
        setTeachers(ts);
        // Prefill workspace from the latest attempt so iterating on a
        // solution doesn't require re-tapping every move.
        const last = a[a.length - 1];
        if (last) setMoves(last.moves.map((m) => ({ x: m.col, y: m.row })));
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Sibling list for prev/next nav. Reload when the source changes.
  useEffect(() => {
    let cancelled = false;
    api.tsumego.listProblems(source)
      .then((ps) => !cancelled && setSiblings(ps))
      .catch(() => !cancelled && setSiblings([]));
    return () => { cancelled = true; };
  }, [source]);

  const navIndex = useMemo(() => {
    if (!siblings) return { idx: -1, prev: null, next: null };
    const idx = siblings.findIndex((p) => p.id === id);
    return {
      idx,
      prev: idx > 0 ? siblings[idx - 1] : null,
      next: idx >= 0 && idx < siblings.length - 1 ? siblings[idx + 1] : null,
    };
  }, [siblings, id]);

  const stones: Stone[] = useMemo(() => {
    return (problem?.stones ?? []).map((s) => ({
      x: s.col, y: s.row, color: s.color as 'B' | 'W',
    }));
  }, [problem]);

  const overlay = useMemo(() => computeNumberedOverlay(moves), [moves]);

  const onPlay = (x: number, y: number) => {
    setMoves((prev) => [...prev, { x, y }]);
  };

  const undo = () => setMoves((prev) => prev.slice(0, -1));
  const clear = () => setMoves([]);

  const saveAttempt = async () => {
    const a = await api.study.submitAttempt(
      id, moves.map((m) => ({ col: m.x, row: m.y })),
    );
    const refreshed = await api.study.listAttempts(id);
    setAttempts(refreshed);
    return a;
  };

  const navSuffix = fromSubmission
    ? `?from_submission=${encodeURIComponent(fromSubmission)}`
    : '';

  const goToNext = () => {
    if (navIndex.next) {
      navigate(`/collections/${encodeURIComponent(source)}/solve/${navIndex.next.id}${navSuffix}`);
    }
  };

  const saveAndContinue = async () => {
    if (moves.length === 0) return;
    setSubmitting(true);
    setError(null);
    setFlash(null);
    try {
      const a = await saveAttempt();
      if (navIndex.next) {
        goToNext();
      } else {
        setFlash(`Saved for submission at ${formatTimestamp(a.submitted_at)}.`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const skip = () => {
    if (!navIndex.next) return;
    setFlash(null);
    setError(null);
    goToNext();
  };

  const backToHome = () => navigate('/');
  const goPrev = () => {
    if (navIndex.prev) {
      navigate(`/collections/${encodeURIComponent(source)}/solve/${navIndex.prev.id}${navSuffix}`);
    }
  };

  if (error && !problem) {
    return (
      <div className="solve">
        <p className="solve-error">{error}</p>
        <p><Link to="/">← home</Link></p>
      </div>
    );
  }
  if (!problem || attempts === null) {
    return <div className="solve" style={{ color: '#666' }}>Loading…</div>;
  }

  // Tight viewport around stones+moves so the working area is large on iPad.
  const allPts = [
    ...stones.map((s) => ({ x: s.x, y: s.y })),
    ...moves,
  ];
  const viewport = boundingViewport(allPts);

  return (
    <div className="solve">
      <header className="solve-header">
        <div>
          {fromSubmission ? (
            <Link
              to={`/submissions/${encodeURIComponent(fromSubmission)}`}
              className="back-link"
            >
              ← back to submission
            </Link>
          ) : (
            <Link
              to={`/collections/${encodeURIComponent(source)}/solve`}
              className="back-link"
            >
              ← {source}
            </Link>
          )}
          <h1>
            Problem {problem.source_board_idx + 1}
            {siblings && navIndex.idx >= 0 && (
              <span className="solve-counter">
                {' '}({navIndex.idx + 1} / {siblings.length})
              </span>
            )}
          </h1>
          <div className="solve-meta">
            {problem.black_to_play ? 'Black to play' : 'White to play'}
          </div>
        </div>
        <div className="solve-nav">
          <button onClick={goPrev} disabled={!navIndex.prev} aria-label="Previous problem">‹ Prev</button>
          <button
            onClick={saveAndContinue}
            disabled={moves.length === 0 || submitting}
            className="solve-save-continue"
            aria-label="Save and continue to next problem"
          >
            {submitting ? 'Saving…' : 'Save & continue ›'}
          </button>
          <button
            onClick={skip}
            disabled={!navIndex.next || submitting}
            aria-label="Skip to next problem without saving"
          >
            Skip ›
          </button>
          <button onClick={backToHome} className="solve-done">Done</button>
        </div>
      </header>

      <div className="solve-workspace">
        <div className="solve-board">
          <Board
            stones={stones}
            numberedMoves={overlay.boardNumbers}
            onPlay={onPlay}
            editable
            displayOnly
            viewport={viewport}
          />
        </div>
        {overlay.chains.length > 0 && (
          <aside className="solve-chains" aria-label="Move sequence">
            <div className="solve-chains-title">Sequence</div>
            <ol className="solve-chains-list">
              {overlay.chains.map((chain, i) => (
                <li key={i} className="chain">
                  {chain.map((n, j) => (
                    <span key={j}>
                      {j > 0 && <span className="chain-sep">→</span>}
                      <span className="chain-num">{n}</span>
                    </span>
                  ))}
                </li>
              ))}
            </ol>
          </aside>
        )}
      </div>

      <div className="solve-info">
        <span>{moves.length} move{moves.length === 1 ? '' : 's'}</span>
        <div className="solve-actions">
          <button onClick={undo} disabled={moves.length === 0 || submitting}>Undo</button>
          <button onClick={clear} disabled={moves.length === 0 || submitting}>Clear</button>
        </div>
      </div>

      {error && <div className="solve-error">{error}</div>}
      {flash && <div className="solve-flash">{flash}</div>}

      <SolveHistory attempts={attempts} teachers={teachers} stones={stones} />

      {problem.image && (
        <details className="solve-original">
          <summary>View original</summary>
          <img
            src={api.tsumego.imageUrl(problem.id)}
            alt={`Original crop for problem ${problem.source_board_idx + 1}`}
          />
        </details>
      )}
    </div>
  );
}

function SolveHistory({
  attempts, teachers, stones,
}: {
  attempts: Attempt[];
  teachers: LinkedUser[];
  stones: Stone[];
}) {
  const teachersById = useMemo(() => {
    const m = new Map<string, LinkedUser>();
    for (const t of teachers) m.set(t.user_id, t);
    return m;
  }, [teachers]);

  const reviewed = useMemo(() => {
    const withReviews = attempts.filter((a) => Object.keys(a.reviews ?? {}).length > 0);
    const latestReview = (a: Attempt) =>
      Object.values(a.reviews).reduce(
        (acc, r) => (r.reviewed_at > acc ? r.reviewed_at : acc), '',
      );
    return [...withReviews].sort((a, b) =>
      latestReview(b).localeCompare(latestReview(a)),
    );
  }, [attempts]);

  if (reviewed.length === 0) return null;

  return (
    <details className="solve-history">
      <summary>
        Submission history ({reviewed.length} attempt{reviewed.length === 1 ? '' : 's'})
      </summary>
      <ul className="solve-history-list">
        {reviewed.map((a) => {
          const moves = a.moves.map((m) => ({ x: m.col, y: m.row }));
          const overlay = computeNumberedOverlay(moves);
          const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
          const viewport = boundingViewport(allPts);
          const verdicts = Object.entries(a.reviews).map(([tid, r]) => ({
            teacher_id: tid,
            label: teachersById.get(tid)?.display_name ?? tid,
            verdict: r.verdict,
            reviewed_at: r.reviewed_at,
          }));
          verdicts.sort((x, y) => y.reviewed_at.localeCompare(x.reviewed_at));
          return (
            <li key={a.id} className="solve-history-row">
              <div className="solve-history-board">
                <Board
                  stones={stones}
                  numberedMoves={overlay.boardNumbers}
                  viewport={viewport}
                  displayOnly
                />
              </div>
              <div className="solve-history-info">
                <div className="solve-history-stamp">
                  {moves.length} move{moves.length === 1 ? '' : 's'}
                  <span className="dot">·</span>
                  submitted {formatTimestamp(a.submitted_at)}
                </div>
                <ul className="solve-history-verdicts">
                  {verdicts.map((v) => (
                    <li key={v.teacher_id} className={`solve-history-verdict v-${v.verdict}`}>
                      <span className="solve-history-mark">
                        {v.verdict === 'correct' ? '✓' : '✗'}
                      </span>
                      <span className="solve-history-label">{v.label}</span>
                      <span className="solve-history-when">
                        {formatTimestamp(v.reviewed_at)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

function boundingViewport(points: { x: number; y: number }[]) {
  if (points.length === 0) {
    return { colMin: 0, colMax: 18, rowMin: 0, rowMax: 18 };
  }
  let xmin = 18, xmax = 0, ymin = 18, ymax = 0;
  for (const p of points) {
    if (p.x < xmin) xmin = p.x;
    if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.y > ymax) ymax = p.y;
  }
  return {
    colMin: Math.max(0, xmin - 3),
    colMax: Math.min(18, xmax + 3),
    rowMin: Math.max(0, ymin - 3),
    rowMax: Math.min(18, ymax + 3),
  };
}

const PAGE_SIZE = 75;

/** Student-facing thumbnail picker for a collection. Tiles link straight
 * into the solver. Editor / QA workflow is reached via the pencil icon
 * on the home page, not from here. Paginated so big collections don't
 * pay for rendering hundreds of Board SVGs up front. */
export function SolveEntry() {
  const { source: encSource = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [problems, setProblems] = useState<TsumegoProblem[] | null>(null);
  const [statuses, setStatuses] = useState<Record<string, ProblemStatus>>({});
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setPage(0);
    Promise.all([
      api.tsumego.listProblems(source),
      api.study.problemStatuses().catch(() => ({} as Record<string, ProblemStatus>)),
    ])
      .then(([ps, st]) => {
        if (cancelled) return;
        setProblems(ps);
        setStatuses(st);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => { cancelled = true; };
  }, [source]);

  if (error) {
    return (
      <div className="solve-picker">
        <p className="solve-error">{error}</p>
        <p><Link to="/">← home</Link></p>
      </div>
    );
  }
  if (problems === null) {
    return <div className="solve-picker" style={{ color: '#666' }}>Loading…</div>;
  }

  const sorted = [...problems].sort((a, b) => a.source_board_idx - b.source_board_idx);
  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * PAGE_SIZE;
  const visible = sorted.slice(start, start + PAGE_SIZE);

  const nextUnattempted = sorted.find((p) => !(p.id in statuses));
  const nextIncorrect = sorted.find((p) => statuses[p.id]?.last_verdict === 'incorrect');
  const goSolve = (id: string) =>
    navigate(`/collections/${encodeURIComponent(source)}/solve/${id}`);

  return (
    <div className="solve-picker">
      <header className="solve-picker-header">
        <Link to="/" className="back-link">← home</Link>
        <h1>{source}</h1>
        <div className="solve-picker-meta">
          {sorted.length === 0
            ? 'No problems in this collection.'
            : `${sorted.length} problem${sorted.length === 1 ? '' : 's'}`}
        </div>
        {sorted.length > 0 && (
          <div className="solve-picker-actions">
            <button
              type="button"
              onClick={() => nextUnattempted && goSolve(nextUnattempted.id)}
              disabled={!nextUnattempted}
            >
              Next problem
            </button>
            <button
              type="button"
              onClick={() => nextIncorrect && goSolve(nextIncorrect.id)}
              disabled={!nextIncorrect}
            >
              Next incorrect problem
            </button>
          </div>
        )}
      </header>

      {sorted.length > 0 && (
        <>
          <ul className="solve-picker-grid">
            {visible.map((p) => (
              <PickerTile
                key={p.id}
                problem={p}
                source={source}
                status={statuses[p.id] ?? null}
              />
            ))}
          </ul>
          {pageCount > 1 && (
            <div className="solve-picker-pager">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={clampedPage === 0}
              >
                ‹ Prev
              </button>
              <span className="solve-picker-page-num">
                Page {clampedPage + 1} of {pageCount}
              </span>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                disabled={clampedPage >= pageCount - 1}
              >
                Next ›
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function PickerTile({
  problem, source, status,
}: {
  problem: TsumegoProblem;
  source: string;
  status: ProblemStatus | null;
}) {
  const stones: Stone[] = (problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const viewport = boundingViewport(stones.map((s) => ({ x: s.x, y: s.y })));
  const verdict = status?.last_verdict ?? null;
  const badge = status
    ? {
        cls: verdict ?? 'pending',
        mark: verdict === 'correct' ? '✓' : verdict === 'incorrect' ? '✗' : '?',
        label: verdict === 'correct' ? 'last attempt correct'
          : verdict === 'incorrect' ? 'last attempt incorrect'
          : 'attempted, not yet graded',
      }
    : null;
  return (
    <li className="solve-picker-tile">
      <Link to={`/collections/${encodeURIComponent(source)}/solve/${problem.id}`}>
        <div className="solve-picker-thumb">
          {stones.length > 0 ? (
            <Board stones={stones} viewport={viewport} displayOnly />
          ) : (
            <div className="solve-picker-noimg">no stones</div>
          )}
          {badge && (
            <div className={`solve-picker-badge badge-${badge.cls}`} aria-label={badge.label}>
              {badge.mark}
            </div>
          )}
        </div>
        <div className="solve-picker-caption">
          #{problem.source_board_idx + 1}
        </div>
      </Link>
    </li>
  );
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
