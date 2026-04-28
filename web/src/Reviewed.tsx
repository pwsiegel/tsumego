import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Board } from './Board';
import {
  api,
  type AttemptWithProblem,
  type LinkedUser,
  type ProblemStatus,
} from './api';
import { computeNumberedOverlay } from './numberedMoves';
import type { Stone } from './types';
import './Reviewed.css';

type ViewMode = 'grouped' | 'flat';

/** Student-facing graded history: latest reviewed attempt per problem,
 * with the verdict from each teacher who weighed in. Click a row to
 * jump back into the solver. Defaults to grouping by submission batch
 * (`sent_at`); the "View all" toggle flattens into one ungrouped list. */
export function Reviewed() {
  const [items, setItems] = useState<AttemptWithProblem[] | null>(null);
  const [teachers, setTeachers] = useState<LinkedUser[] | null>(null);
  const [statuses, setStatuses] = useState<Record<string, ProblemStatus>>({});
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>('grouped');

  useEffect(() => {
    Promise.all([api.study.listReviewed(), api.study.listTeachers()])
      .then(([its, ts]) => { setItems(its); setTeachers(ts); })
      .catch((e) => setError(String(e)));
    api.study.problemStatuses()
      .then(setStatuses)
      .catch(() => {});
  }, []);

  const teachersById = useMemo(() => {
    const m = new Map<string, LinkedUser>();
    for (const t of teachers ?? []) m.set(t.user_id, t);
    return m;
  }, [teachers]);

  const groups = useMemo(() => groupBySentAt(items ?? []), [items]);

  const unretriedIncorrect = useMemo(() => {
    return (items ?? []).filter((it) => {
      const reviews = Object.values(it.attempt.reviews);
      if (reviews.length === 0) return false;
      const latest = reviews.reduce(
        (a, b) => (a.reviewed_at > b.reviewed_at ? a : b),
      );
      if (latest.verdict !== 'incorrect') return false;
      const latestAt = statuses[it.problem.id]?.latest_attempt_at;
      return !(latestAt != null && latestAt > it.attempt.submitted_at);
    });
  }, [items, statuses]);

  const retryHref = unretriedIncorrect.length > 0
    ? `/collections/${encodeURIComponent(unretriedIncorrect[0].problem.source)}/solve/${unretriedIncorrect[0].problem.id}?retry=incorrect`
    : null;

  if (items === null || teachers === null) {
    return (
      <div className="reviewed">
        <p className="dim">{error ?? 'Loading…'}</p>
      </div>
    );
  }

  return (
    <div className="reviewed">
      <header className="reviewed-header">
        <Link to="/" className="back-link">← home</Link>
        <h1>Submission history</h1>
        <div className="reviewed-meta">
          {items.length === 0
            ? 'Nothing graded yet.'
            : `${items.length} problem${items.length === 1 ? '' : 's'} reviewed.`}
        </div>
        {items.length > 0 && (
          <div className="reviewed-header-actions">
            <div className="reviewed-toggle">
              <button
                type="button"
                className={`reviewed-toggle-btn${mode === 'grouped' ? ' active' : ''}`}
                onClick={() => setMode('grouped')}
              >
                By submission
              </button>
              <button
                type="button"
                className={`reviewed-toggle-btn${mode === 'flat' ? ' active' : ''}`}
                onClick={() => setMode('flat')}
              >
                View all
              </button>
            </div>
            {retryHref && (
              <Link to={retryHref} className="submission-retry-btn">
                Retry incorrect ({unretriedIncorrect.length})
              </Link>
            )}
          </div>
        )}
      </header>

      {error && <div className="reviewed-error">{error}</div>}

      {items.length > 0 && mode === 'flat' && (
        <ul className="reviewed-list">
          {items.map((it) => (
            <ReviewedRow
              key={it.attempt.id}
              item={it}
              teachersById={teachersById}
              status={statuses[it.problem.id] ?? null}
            />
          ))}
        </ul>
      )}

      {items.length > 0 && mode === 'grouped' && (
        <div className="reviewed-groups">
          {groups.map((g) => (
            <section key={g.sent_at || 'unsent'} className="reviewed-group">
              <h2 className="reviewed-group-header">
                {g.sent_at
                  ? `Submitted ${formatTimestamp(g.sent_at)}`
                  : 'Submitted (unknown date)'}
                <span className="reviewed-group-count">
                  {g.items.length} problem{g.items.length === 1 ? '' : 's'}
                </span>
              </h2>
              <ul className="reviewed-list">
                {g.items.map((it) => (
                  <ReviewedRow
                    key={it.attempt.id}
                    item={it}
                    teachersById={teachersById}
                    status={statuses[it.problem.id] ?? null}
                  />
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function groupBySentAt(items: AttemptWithProblem[]): {
  sent_at: string;
  items: AttemptWithProblem[];
}[] {
  const by = new Map<string, AttemptWithProblem[]>();
  for (const it of items) {
    const k = it.attempt.sent_at ?? '';
    if (!by.has(k)) by.set(k, []);
    by.get(k)!.push(it);
  }
  const out = Array.from(by.entries()).map(([sent_at, items]) => ({ sent_at, items }));
  out.sort((a, b) => (b.sent_at || '').localeCompare(a.sent_at || ''));
  return out;
}

function ReviewedRow({
  item, teachersById, status,
}: {
  item: AttemptWithProblem;
  teachersById: Map<string, LinkedUser>;
  status: ProblemStatus | null;
}) {
  const stones: Stone[] = (item.problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const moves = item.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(moves);
  const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
  const viewport = boundingViewport(allPts);

  const reviews = Object.entries(item.attempt.reviews).map(([tid, r]) => ({
    teacher_id: tid,
    label: teachersById.get(tid)?.display_name ?? tid,
    verdict: r.verdict,
    reviewed_at: r.reviewed_at,
  }));
  // Newest verdict first.
  reviews.sort((a, b) => b.reviewed_at.localeCompare(a.reviewed_at));

  const solveHref = `/collections/${encodeURIComponent(item.problem.source)}/solve/${item.problem.id}`;
  const retried = status?.latest_attempt_at != null
    && status.latest_attempt_at > item.attempt.submitted_at;

  return (
    <li className="reviewed-row">
      <Link to={solveHref} className="reviewed-row-link" aria-label="Open in solver">
        <div className="reviewed-row-board">
          <Board
            stones={stones}
            numberedMoves={overlay.boardNumbers}
            viewport={viewport}
            displayOnly
          />
        </div>
        <div className="reviewed-row-info">
          <div className="reviewed-row-source">
            <span className="reviewed-row-collection">{item.problem.source}</span>
            <span className="dot">·</span>
            <span>Problem {item.problem.source_board_idx + 1}</span>
          </div>
          <div className="reviewed-row-stamp">
            {moves.length} move{moves.length === 1 ? '' : 's'}
            <span className="dot">·</span>
            submitted {formatTimestamp(item.attempt.submitted_at)}
          </div>
          <ul className="reviewed-verdicts">
            {reviews.map((r) => (
              <li
                key={r.teacher_id}
                className={
                  retried && r.verdict === 'incorrect'
                    ? 'reviewed-verdict v-incorrect-retried'
                    : `reviewed-verdict v-${r.verdict}`
                }
              >
                <span className="reviewed-verdict-mark">
                  {r.verdict === 'correct' ? '✓' : '✗'}
                </span>
                <span className="reviewed-verdict-label">{r.label}</span>
                <span className="reviewed-verdict-when">
                  {formatTimestamp(r.reviewed_at)}
                </span>
              </li>
            ))}
            {retried && reviews.some((r) => r.verdict === 'incorrect') && (
              <li className="reviewed-verdict v-retried">
                <span className="reviewed-verdict-mark">↻</span>
                <span className="reviewed-verdict-label">incorrect but retried</span>
              </li>
            )}
          </ul>
        </div>
      </Link>
    </li>
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
    colMin: Math.max(0, xmin - 1),
    colMax: Math.min(18, xmax + 1),
    rowMin: Math.max(0, ymin - 1),
    rowMax: Math.min(18, ymax + 1),
  };
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
