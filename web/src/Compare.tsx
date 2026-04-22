import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Board } from './Board';
import type { Stone } from './types';
import './Compare.css';

type ComparisonStone = { col: number; row: number; color: string };

type ComparisonProblem = {
  stem: string;
  source_board_idx: number;
  crop_width: number;
  crop_height: number;
  gt: ComparisonStone[];
  old: ComparisonStone[];
  new: ComparisonStone[];
  old_matches_gt: boolean;
  new_matches_gt: boolean;
};

type ComparisonData = {
  val_dir: string;
  old_model: string;
  new_model: string;
  total: number;
  changed_count: number;
  problems: ComparisonProblem[];
};

export function Compare() {
  const { dataset = 'hm2' } = useParams();
  const [data, setData] = useState<ComparisonData | null>(null);
  const [idx, setIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [editedGt, setEditedGt] = useState<ComparisonStone[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [editCount, setEditCount] = useState<number>(0);

  const refreshEditCount = async () => {
    try {
      const r = await fetch(`/api/val/${dataset}/gt-edits`, { cache: 'no-store' });
      if (r.ok) {
        const edits = await r.json();
        setEditCount(Array.isArray(edits) ? edits.length : 0);
      }
    } catch { /* best-effort */ }
  };

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`/api/val/${dataset}/comparison`, { cache: 'no-store' });
        if (!r.ok) throw new Error(r.statusText);
        setData(await r.json());
      } catch (e) {
        setError(String(e));
      }
    })();
    refreshEditCount();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  // Reset any unsaved GT edits when the selected problem changes.
  useEffect(() => { setEditedGt(null); }, [idx]);

  // Guard nav when there are unsaved GT edits.
  const maybeNav = (dir: -1 | 1) => {
    if (editedGt && !confirm('You have unsaved ground-truth edits. Discard and navigate?')) {
      return;
    }
    setIdx((i) => {
      if (!data) return i;
      if (dir === -1) return Math.max(0, i - 1);
      return Math.min(data.problems.length - 1, i + 1);
    });
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!data) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); maybeNav(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); maybeNav(1); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, editedGt]);

  if (error) return <div className="compare-msg">Error: {error}</div>;
  if (!data) return <div className="compare-msg">Loading…</div>;
  if (data.problems.length === 0) {
    return <div className="compare-msg">No problems differ between models — nothing to show.</div>;
  }

  const p = data.problems[idx];
  const currentGt = editedGt ?? p.gt;
  const toStones = (ss: ComparisonStone[]): Stone[] =>
    ss.map((s) => ({ x: s.col, y: s.row, color: s.color === 'B' ? 'B' : 'W' }));

  const sameSet = (a: ComparisonStone[], b: ComparisonStone[]): boolean => {
    if (a.length !== b.length) return false;
    const key = (s: ComparisonStone) => `${s.col},${s.row},${s.color}`;
    const A = new Set(a.map(key));
    return b.every((s) => A.has(key(s)));
  };

  const dirty = editedGt !== null && !sameSet(editedGt, p.gt);

  // Same click semantics as ProblemEditor: click = B or remove,
  // shift-click = W or remove. Removing always works regardless of color.
  const handleGtClick = (x: number, y: number, shift: boolean) => {
    const base = editedGt ?? p.gt;
    const existing = base.find((s) => s.col === x && s.row === y);
    let next: ComparisonStone[];
    if (existing) next = base.filter((s) => !(s.col === x && s.row === y));
    else next = [...base, { col: x, row: y, color: shift ? 'W' : 'B' }];
    setEditedGt(next);
  };

  const saveGt = async () => {
    if (!editedGt || !data) return;
    setSaving(true);
    try {
      const r = await fetch(
        `/api/val/${dataset}/problems/${p.stem}/stones`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stones: editedGt }),
        },
      );
      if (!r.ok) {
        const detail = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(detail.detail ?? r.statusText);
      }
      // Update local data to reflect the save: replace gt, recompute matches.
      const oldSet = new Set(p.old.map((s) => `${s.col},${s.row},${s.color}`));
      const newSet = new Set(p.new.map((s) => `${s.col},${s.row},${s.color}`));
      const gtSet = new Set(editedGt.map((s) => `${s.col},${s.row},${s.color}`));
      const setsEq = (a: Set<string>, b: Set<string>) =>
        a.size === b.size && [...a].every((x) => b.has(x));
      const updated = { ...data };
      updated.problems = data.problems.map((prob, i) =>
        i === idx ? {
          ...prob,
          gt: editedGt,
          old_matches_gt: setsEq(oldSet, gtSet),
          new_matches_gt: setsEq(newSet, gtSet),
        } : prob
      );
      setData(updated);
      setEditedGt(null);
      refreshEditCount();
    } catch (e) {
      alert(`Save failed: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  const resetGt = () => setEditedGt(null);

  // Use the same 3-cell margin viewport computed from ALL stones visible
  // on this problem, so the three panels render the same region.
  const allStones = [...currentGt, ...p.old, ...p.new];
  let viewport: { colMin: number; colMax: number; rowMin: number; rowMax: number } | undefined;
  if (allStones.length > 0) {
    const cols = allStones.map((s) => s.col);
    const rows = allStones.map((s) => s.row);
    viewport = {
      colMin: Math.max(0, Math.min(...cols) - 3),
      colMax: Math.min(18, Math.max(...cols) + 3),
      rowMin: Math.max(0, Math.min(...rows) - 3),
      rowMax: Math.min(18, Math.max(...rows) + 3),
    };
  }

  // Diff annotations: when comparing against the CURRENT (possibly edited)
  // GT, show which stones each model has that the GT doesn't.
  const gtSet = new Set(currentGt.map((s) => `${s.col},${s.row},${s.color}`));
  const newDiffs = p.new.filter((s) => !gtSet.has(`${s.col},${s.row},${s.color}`));
  const oldDiffs = p.old.filter((s) => !gtSet.has(`${s.col},${s.row},${s.color}`));
  const oldMatches = oldDiffs.length === 0 && p.old.length === currentGt.length;
  const newMatches = newDiffs.length === 0 && p.new.length === currentGt.length;

  return (
    <div className="compare">
      <header className="compare-header">
        <div>
          <Link to="/" className="back">← home</Link>
          <h1>Model comparison: {dataset}</h1>
          <div className="compare-sub">
            {data.problems.length} problems where old ≠ new &nbsp;·&nbsp;
            old matches GT: {data.problems.filter((x) => x.old_matches_gt).length} &nbsp;·&nbsp;
            new matches GT: {data.problems.filter((x) => x.new_matches_gt).length} &nbsp;·&nbsp;
            old model: <code>{data.old_model.split('/').pop()}</code> &nbsp;vs&nbsp;
            new model: <code>{data.new_model.split('/').pop()}</code>
            {editCount > 0 && (
              <> &nbsp;·&nbsp; <a href={`/api/val/${dataset}/gt-edits`} target="_blank" rel="noreferrer">
                {editCount} GT {editCount === 1 ? 'edit' : 'edits'} saved
              </a></>
            )}
          </div>
        </div>
        <div className="compare-nav">
          <button onClick={() => maybeNav(-1)} disabled={idx === 0}>◀</button>
          <span className="compare-idx">{idx + 1} / {data.problems.length}</span>
          <button
            onClick={() => maybeNav(1)}
            disabled={idx === data.problems.length - 1}
          >▶</button>
        </div>
      </header>

      <div className="compare-which">
        <select
          value={idx}
          onChange={(e) => setIdx(parseInt(e.target.value, 10))}
        >
          {data.problems.map((prob, i) => (
            <option key={prob.stem} value={i}>
              {prob.stem} (board #{prob.source_board_idx + 1}) &nbsp;
              {prob.old_matches_gt ? '· old=GT' : ''}
              {prob.new_matches_gt ? '· new=GT' : ''}
              {!prob.old_matches_gt && !prob.new_matches_gt ? '· neither' : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="compare-grid">
        <div className="compare-panel">
          <div className="compare-label">Original crop</div>
          <img
            className="compare-crop"
            src={`/api/val/${dataset}/images/${p.stem}.png`}
            alt={p.stem}
          />
        </div>

        <div className="compare-panel">
          <div className={`compare-label ${oldMatches ? 'match' : 'diff'}`}>
            Old model ({p.old.length} stones)
            {oldMatches && ' · matches GT'}
            {!oldMatches && oldDiffs.length > 0 && (
              <> · {oldDiffs.length} {oldDiffs.length === 1 ? 'diff' : 'diffs'}</>
            )}
          </div>
          <Board stones={toStones(p.old)} onPlay={() => {}} viewport={viewport} displayOnly showCoords />
        </div>

        <div className="compare-panel">
          <div className={`compare-label ${newMatches ? 'match' : 'diff'}`}>
            New model ({p.new.length} stones)
            {newMatches && ' · matches GT'}
            {!newMatches && newDiffs.length > 0 && (
              <> · {newDiffs.length} {newDiffs.length === 1 ? 'diff' : 'diffs'}</>
            )}
          </div>
          <Board stones={toStones(p.new)} onPlay={() => {}} viewport={viewport} displayOnly showCoords />
        </div>

        <div className="compare-panel">
          <div className={`compare-label gt${dirty ? ' dirty' : ''}`}>
            Ground truth ({currentGt.length} stones)
            {dirty && ' · unsaved edits'}
          </div>
          <Board
            stones={toStones(currentGt)}
            onPlay={handleGtClick}
            viewport={viewport}
            editable
            showCoords
          />
          <div className="gt-actions">
            <button onClick={resetGt} disabled={!dirty || saving} className="gt-reset">
              Reset
            </button>
            <button onClick={saveGt} disabled={!dirty || saving} className="gt-save">
              {saving ? 'Saving…' : 'Save GT'}
            </button>
          </div>
        </div>
      </div>

      {(oldDiffs.length > 0 || newDiffs.length > 0) && (
        <div className="compare-diff-list">
          {oldDiffs.length > 0 && (
            <div>
              <strong>Old vs GT differences:</strong>{' '}
              {oldDiffs.map((s, i) => (
                <code key={i}>
                  {s.color}@({s.col},{s.row})
                  {i < oldDiffs.length - 1 ? ' ' : ''}
                </code>
              ))}
            </div>
          )}
          {newDiffs.length > 0 && (
            <div>
              <strong>New vs GT differences:</strong>{' '}
              {newDiffs.map((s, i) => (
                <code key={i}>
                  {s.color}@({s.col},{s.row})
                  {i < newDiffs.length - 1 ? ' ' : ''}
                </code>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
