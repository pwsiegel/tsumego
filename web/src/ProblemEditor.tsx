import { useEffect, useRef, useState } from 'react';
import { Board } from './Board';
import { api, type TsumegoProblem } from './api';
import type { Stone } from './types';
import './ProblemEditor.css';

type EditorStone = Stone;

type DecisionKind = 'accepted' | 'rejected';

type Props = {
  problem: TsumegoProblem;
  /** Called after the server confirms the decision. Receives the final
   * status ("accepted" | "accepted_edited" | "rejected"). */
  onDecision: (status: string) => void;
  /** Optional back button target in the header. */
  onExit?: () => void;
  /** Show ← and → nav buttons if provided. */
  onPrev?: () => void;
  onNext?: () => void;
  /** Progress label, e.g. "Problem 3 of 12". */
  label?: string;
};

export function ProblemEditor({
  problem, onDecision, onExit, onPrev, onNext, label,
}: Props) {
  const toEditor = (ss: TsumegoProblem['stones']): EditorStone[] => {
    // Dedupe by (col, row) (discretizer can snap two detections to the same cell).
    const byCell = new Map<string, EditorStone>();
    for (const s of ss) {
      const key = `${s.col},${s.row}`;
      if (byCell.has(key)) continue;
      byCell.set(key, {
        x: s.col, y: s.row, color: s.color === 'B' ? 'B' : 'W',
      });
    }
    return Array.from(byCell.values());
  };

  const [originalStones] = useState<EditorStone[]>(
    () => toEditor(problem.stones)
  );
  const [editedStones, setEditedStones] = useState<EditorStone[]>(
    () => toEditor(problem.stones)
  );
  const [status, setStatus] = useState(problem.status);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const dirty = editedStones.length !== originalStones.length
    || editedStones.some((s) => {
      const o = originalStones.find((x) => x.x === s.x && x.y === s.y);
      return !o || o.color !== s.color;
    });

  // Click: place B (or remove existing). Shift-click: place W (or remove).
  // Clicking an occupied intersection always removes the stone, regardless
  // of which modifier. Covers the "oops, this shouldn't be a stone" case
  // without a detour through the other color.
  const handleIntersectionClick = (x: number, y: number, shift: boolean) => {
    setEditedStones((prev) => {
      const existing = prev.find((s) => s.x === x && s.y === y);
      if (existing) return prev.filter((s) => !(s.x === x && s.y === y));
      return [...prev, { x, y, color: shift ? 'W' : 'B' }];
    });
  };

  const saveDecision = async (decision: DecisionKind) => {
    setSaving(true);
    try {
      // "accepted" with edits → "accepted_edited" so we can flag these
      // later. Rejects always save the user's current stones too.
      let finalStatus: string = decision;
      if (decision === 'accepted' && dirty) finalStatus = 'accepted_edited';
      await api.tsumego.updateProblem(problem.id, {
        status: finalStatus,
        stones: editedStones.map((s) => ({ col: s.x, row: s.y, color: s.color })),
        black_to_play: problem.black_to_play,
      });
      setStatus(finalStatus);
      onDecision(finalStatus);
    } catch (e) {
      setMessage(`Save failed: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  // Keyboard shortcuts: ← prev, → next, Enter accept. The ref tracks the
  // latest saveDecision closure so the keydown handler always sees fresh
  // editedStones/dirty without re-binding the listener every render.
  const acceptRef = useRef<() => void>(() => {});
  useEffect(() => {
    acceptRef.current = () => saveDecision('accepted');
  });
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (e.key === 'ArrowLeft' && onPrev) { e.preventDefault(); onPrev(); }
      else if (e.key === 'ArrowRight' && onNext) { e.preventDefault(); onNext(); }
      else if (e.key === 'Enter') { e.preventDefault(); acceptRef.current(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onPrev, onNext]);

  // Crop the 19×19 view to ~3 cells of margin around the stones.
  const MARGIN = 3;
  const allStones = [...originalStones, ...editedStones];
  let viewport: { colMin: number; colMax: number; rowMin: number; rowMax: number } | undefined;
  if (allStones.length > 0) {
    const cols = allStones.map((s) => s.x);
    const rows = allStones.map((s) => s.y);
    viewport = {
      colMin: Math.max(0, Math.min(...cols) - MARGIN),
      colMax: Math.min(18, Math.max(...cols) + MARGIN),
      rowMin: Math.max(0, Math.min(...rows) - MARGIN),
      rowMax: Math.min(18, Math.max(...rows) + MARGIN),
    };
  }

  const reset = () => setEditedStones(originalStones);
  const clearBoard = () => setEditedStones([]);

  return (
    <div className="editor-page">
      <header className="editor-header">
        <h1>
          {problem.source}
          {label && <span className="editor-label"> &nbsp;·&nbsp; {label}</span>}
        </h1>
        <nav>{onExit && <a href="#" onClick={(e) => { e.preventDefault(); onExit(); }}>back</a>}</nav>
      </header>

      {message && <div className="editor-message">{message}</div>}

      <div className="editor-stage">
        <div className={`editor-crop status-${status}`}>
          {problem.image && (
            <img
              src={api.tsumego.imageUrl(problem.id)}
              alt={`problem ${problem.source_board_idx}`}
            />
          )}
          {status === 'accepted' && <DecisionOverlay kind="accepted" />}
          {status === 'accepted_edited' && <DecisionOverlay kind="accepted_edited" />}
          {status === 'rejected' && <DecisionOverlay kind="rejected" />}
        </div>
        <div className="editor-board">
          <Board
            stones={editedStones}
            onPlay={handleIntersectionClick}
            editable
            viewport={viewport}
          />
        </div>
      </div>

      <div className="editor-actions">
        {onPrev && (
          <button
            className="btn-nav"
            onClick={onPrev}
            title="Previous (←)"
            aria-label="Previous problem"
          >◀</button>
        )}
        <button
          className="btn-reject"
          onClick={() => saveDecision('rejected')}
          disabled={saving}
        >
          Reject
        </button>
        <button
          className="btn-reset"
          onClick={reset}
          disabled={saving || !dirty}
          title="Revert to detected stones"
        >
          Reset
        </button>
        <button
          className="btn-clear"
          onClick={clearBoard}
          disabled={saving || editedStones.length === 0}
          title="Remove all stones"
        >
          Clear
        </button>
        <button
          className="btn-accept"
          onClick={() => saveDecision('accepted')}
          disabled={saving}
          title="Accept (Enter)"
        >
          {saving ? 'Saving…' : dirty ? 'Accept (with edits)' : 'Accept'}
        </button>
        {onNext && (
          <button
            className="btn-nav"
            onClick={onNext}
            title="Next (→)"
            aria-label="Next problem"
          >▶</button>
        )}
      </div>
    </div>
  );
}


function DecisionOverlay({ kind }: { kind: 'accepted' | 'accepted_edited' | 'rejected' }) {
  const pathAccepted = 'M20 52 L44 76 L82 28';
  const pathReject = 'M22 22 L78 78 M78 22 L22 78';
  const accepted = kind === 'accepted' || kind === 'accepted_edited';
  return (
    <div className={`status-overlay status-${kind}`} aria-label={kind}>
      <svg viewBox="0 0 100 100" width="100" height="100">
        <path
          d={accepted ? pathAccepted : pathReject}
          stroke="currentColor"
          strokeWidth="14"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
    </div>
  );
}
