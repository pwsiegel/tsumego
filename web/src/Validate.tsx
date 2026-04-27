import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Board } from './Board';
import {
  api,
  type Flip,
  type ProblemResult,
  type RunResult,
  type ServerStone,
} from './api';
import type { Stone } from './types';
import './Validate.css';

type RunState =
  | { kind: 'error'; dataset: string; message: string }
  | { kind: 'running'; dataset: string; total: number | null; problems: ProblemResult[] }
  | { kind: 'success'; dataset: string; result: RunResult };

export function Validate() {
  const { dataset } = useParams<{ dataset: string }>();
  const [state, setState] = useState<RunState | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'changed' | 'error'>('changed');

  useEffect(() => {
    if (!dataset) return;
    const ac = new AbortController();
    setState({ kind: 'running', dataset, total: null, problems: [] });

    let total: number | null = null;
    let filterStatus = 'accepted';
    const problems: ProblemResult[] = [];

    api.val.runValidationStream(dataset, (event) => {
      if (ac.signal.aborted) return;
      if (event.event === 'start') {
        total = event.total;
        filterStatus = event.filter_status;
        setState({ kind: 'running', dataset, total, problems: [...problems] });
      } else if (event.event === 'problem') {
        problems.push(event.result);
        setState({ kind: 'running', dataset, total, problems: [...problems] });
      } else if (event.event === 'done') {
        setState({
          kind: 'success', dataset,
          result: {
            dataset, filter_status: filterStatus,
            total: total ?? problems.length,
            exact: event.exact, changed: event.changed, errors: event.errors,
            problems: [...problems],
          },
        });
      }
    }, { signal: ac.signal })
      .catch((e) => {
        if (ac.signal.aborted) return;
        setState({ kind: 'error', dataset, message: String(e) });
      });

    return () => { ac.abort(); };
  }, [dataset]);

  // While the dataset URL param has changed but the new fetch hasn't landed
  // yet, `state` still holds the previous dataset's result — render loading.
  if (!state || state.dataset !== dataset) {
    return (
      <div className="validate">
        <h1>Validation: {dataset}</h1>
        <p className="loading">Starting pipeline...</p>
      </div>
    );
  }

  if (state.kind === 'error') {
    return (
      <div className="validate">
        <h1>Validation: {dataset}</h1>
        <p className="error">Error: {state.message}</p>
      </div>
    );
  }

  if (state.kind === 'running') {
    const done = state.problems.length;
    const total = state.total;
    const pct = total && total > 0 ? Math.round((done / total) * 100) : 0;
    return (
      <div className="validate">
        <h1>Validation: {dataset}</h1>
        <div className="progress-wrap">
          <div className="progress-label">
            {total === null
              ? 'Starting...'
              : `Processed ${done} / ${total} (${pct}%)`}
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
    );
  }

  const result = state.result;

  const pct = ((result.exact / result.total) * 100).toFixed(1);
  const filtered = result.problems.filter((p) => {
    if (filter === 'all') return p.status !== 'exact';
    return p.status === filter;
  });

  return (
    <div className="validate">
      <h1>Validation: {dataset}</h1>

      <div className="summary">
        <div className="stat exact">
          <span className="num">{result.exact}</span>
          <span className="label">exact ({pct}%)</span>
        </div>
        <div className="stat changed">
          <span className="num">{result.changed}</span>
          <span className="label">changed</span>
        </div>
        <div className="stat errors">
          <span className="num">{result.errors}</span>
          <span className="label">errors</span>
        </div>
        <div className="stat total">
          <span className="num">{result.total}</span>
          <span className="label">total</span>
        </div>
      </div>

      {result.changed > 0 || result.errors > 0 ? (
        <>
          <div className="filter-bar">
            <span>Show:</span>
            <button
              className={filter === 'changed' ? 'active' : ''}
              onClick={() => setFilter('changed')}
            >
              Changed ({result.changed})
            </button>
            <button
              className={filter === 'error' ? 'active' : ''}
              onClick={() => setFilter('error')}
            >
              Errors ({result.errors})
            </button>
            <button
              className={filter === 'all' ? 'active' : ''}
              onClick={() => setFilter('all')}
            >
              All non-exact
            </button>
          </div>

          <div className="problem-list">
            {filtered.map((p) => (
              <ProblemRow
                key={p.stem}
                problem={p}
                dataset={dataset!}
                isExpanded={expanded === p.stem}
                onToggle={() =>
                  setExpanded(expanded === p.stem ? null : p.stem)
                }
              />
            ))}
            {filtered.length === 0 && (
              <p className="empty">No problems match this filter.</p>
            )}
          </div>
        </>
      ) : (
        <p className="all-good">All problems match exactly.</p>
      )}

      <p className="back-link">
        <Link to="/testing">← back to testing</Link>
      </p>
    </div>
  );
}

function ProblemRow({
  problem: p,
  dataset,
  isExpanded,
  onToggle,
}: {
  problem: ProblemResult;
  dataset: string;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const missedCount = p.missed?.length ?? 0;
  const extraCount = p.extra?.length ?? 0;
  const flipCount = p.flips?.length ?? 0;

  return (
    <div className={`problem-row ${p.status}`}>
      <div className="problem-header" onClick={onToggle}>
        <span className={`badge ${p.status}`}>{p.status}</span>
        <span className="stem">{p.stem}</span>
        {p.status === 'changed' && (
          <span className="diff-summary">
            {missedCount > 0 && (
              <span className="missed">-{missedCount} missed</span>
            )}
            {extraCount > 0 && (
              <span className="extra">+{extraCount} extra</span>
            )}
            {flipCount > 0 && (
              <span className="flipped">{flipCount} flipped</span>
            )}
          </span>
        )}
        {p.status === 'error' && (
          <span className="error-msg">{p.error}</span>
        )}
        <span className="expand-icon">{isExpanded ? '\u25B2' : '\u25BC'}</span>
      </div>

      {isExpanded && p.status === 'changed' && (
        <ProblemDetail problem={p} dataset={dataset} />
      )}
    </div>
  );
}

function ProblemDetail({
  problem: p,
  dataset,
}: {
  problem: ProblemResult;
  dataset: string;
}) {
  // Build stone lists for the two boards. Ground-truth board shows all GT
  // stones with missed ones highlighted by the overlay; predicted board shows
  // all predicted stones with extras highlighted; flips are marked on both.

  // Compute viewport from all stones involved
  const allPositions = [
    ...(p.gt_stones ?? []),
    ...(p.pred_stones ?? []),
  ];
  let viewport = undefined;
  if (allPositions.length > 0) {
    const cols = allPositions.map((s) => s.col);
    const rows = allPositions.map((s) => s.row);
    const pad = 2;
    viewport = {
      colMin: Math.max(0, Math.min(...cols) - pad),
      colMax: Math.min(18, Math.max(...cols) + pad),
      rowMin: Math.max(0, Math.min(...rows) - pad),
      rowMax: Math.min(18, Math.max(...rows) + pad),
    };
  }

  // GT stones with numbers for missed/flipped
  const gtStones: Stone[] = (p.gt_stones ?? []).map((s) => ({
    x: s.col,
    y: s.row,
    color: s.color as 'B' | 'W',
  }));

  const predStones: Stone[] = (p.pred_stones ?? []).map((s) => ({
    x: s.col,
    y: s.row,
    color: s.color as 'B' | 'W',
  }));

  return (
    <div className="problem-detail">
      <div className="detail-boards">
        <div className="detail-col">
          <div className="detail-label">
            Ground truth ({p.gt_count} stones)
          </div>
          <div className="board-wrap">
            <Board
              stones={gtStones}
              onPlay={() => {}}
              displayOnly
              viewport={viewport}
              showCoords
            />
            {/* Overlay markers for missed stones */}
            <DiffOverlay
              missed={p.missed ?? []}
              extra={[]}
              flips={p.flips ?? []}
              viewport={viewport}
            />
          </div>
        </div>
        <div className="detail-col">
          <div className="detail-label">
            Predicted ({p.pred_count} stones)
          </div>
          <div className="board-wrap">
            <Board
              stones={predStones}
              onPlay={() => {}}
              displayOnly
              viewport={viewport}
              showCoords
            />
            <DiffOverlay
              missed={[]}
              extra={p.extra ?? []}
              flips={p.flips ?? []}
              viewport={viewport}
            />
          </div>
        </div>
        <div className="detail-col image-col">
          <div className="detail-label">Crop image</div>
          <img
            src={api.val.imageUrl(dataset, p.stem)}
            alt={p.stem}
            className="crop-image"
          />
        </div>
      </div>

      {(p.missed?.length || p.extra?.length || p.flips?.length) ? (
        <div className="diff-details">
          {p.missed && p.missed.length > 0 && (
            <div className="diff-group">
              <span className="diff-label missed">Missed:</span>
              {p.missed.map((s, i) => (
                <span key={i} className="diff-cell">
                  {s.color}({s.col},{s.row})
                </span>
              ))}
            </div>
          )}
          {p.extra && p.extra.length > 0 && (
            <div className="diff-group">
              <span className="diff-label extra">Extra:</span>
              {p.extra.map((s, i) => (
                <span key={i} className="diff-cell">
                  {s.color}({s.col},{s.row})
                </span>
              ))}
            </div>
          )}
          {p.flips && p.flips.length > 0 && (
            <div className="diff-group">
              <span className="diff-label flipped">Flipped:</span>
              {p.flips.map((f, i) => (
                <span key={i} className="diff-cell">
                  ({f.col},{f.row}) {f.gt_color}→{f.pred_color}
                </span>
              ))}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

// SVG overlay that draws X marks on missed/extra/flipped stones
const PADDING = 30;
const CELL = 32;
const SIZE = PADDING * 2 + CELL * 18;

function toPx(i: number) {
  return PADDING + i * CELL;
}

function DiffOverlay({
  missed,
  extra,
  flips,
  viewport,
}: {
  missed: ServerStone[];
  extra: ServerStone[];
  flips: Flip[];
  viewport?: { colMin: number; colMax: number; rowMin: number; rowMax: number };
}) {
  const BUF = CELL * 0.7;
  let vb = `0 0 ${SIZE} ${SIZE}`;
  if (viewport) {
    const vx0 = viewport.colMin <= 0 ? 0 : toPx(viewport.colMin) - BUF;
    const vy0 = viewport.rowMin <= 0 ? 0 : toPx(viewport.rowMin) - BUF;
    const vx1 = viewport.colMax >= 18 ? SIZE : toPx(viewport.colMax) + BUF;
    const vy1 = viewport.rowMax >= 18 ? SIZE : toPx(viewport.rowMax) + BUF;
    vb = `${vx0} ${vy0} ${vx1 - vx0} ${vy1 - vy0}`;
  }

  const r = CELL * 0.35;
  return (
    <svg className="diff-overlay" viewBox={vb} xmlns="http://www.w3.org/2000/svg">
      {/* Missed: red X on GT board */}
      {missed.map((s, i) => (
        <g key={`m-${i}`}>
          <line
            x1={toPx(s.col) - r} y1={toPx(s.row) - r}
            x2={toPx(s.col) + r} y2={toPx(s.row) + r}
            stroke="#e53935" strokeWidth={3}
          />
          <line
            x1={toPx(s.col) + r} y1={toPx(s.row) - r}
            x2={toPx(s.col) - r} y2={toPx(s.row) + r}
            stroke="#e53935" strokeWidth={3}
          />
        </g>
      ))}
      {/* Extra: red circle on Pred board */}
      {extra.map((s, i) => (
        <circle
          key={`e-${i}`}
          cx={toPx(s.col)} cy={toPx(s.row)}
          r={r}
          fill="none" stroke="#e53935" strokeWidth={3}
        />
      ))}
      {/* Flips: orange diamond */}
      {flips.map((f, i) => (
        <polygon
          key={`f-${i}`}
          points={`${toPx(f.col)},${toPx(f.row) - r} ${toPx(f.col) + r},${toPx(f.row)} ${toPx(f.col)},${toPx(f.row) + r} ${toPx(f.col) - r},${toPx(f.row)}`}
          fill="none" stroke="#ff9800" strokeWidth={3}
        />
      ))}
    </svg>
  );
}
