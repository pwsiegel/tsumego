import { useEffect, useState, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Board } from './Board';
import { ProblemCard } from './ProblemCard';
import { PlayView } from './PlayView';
import { computeNumberedOverlay } from './numberedMoves';
import { toStones, boundingViewport } from './stones';
import { imageUrl } from './data/library';
import { useBackdropDismiss } from './backdrop';
import type { LibProblem, Move, Verdict } from './data/model';
import './ProblemModal.css';
import './Solve.css';

/** Generic modal shell for the route-modal pattern: renders children (a full
 * view) over the background page. Closes on backdrop / × / Escape via history. */
export function ProblemModalShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') navigate(-1); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev; };
  }, [navigate]);
  return (
    <div className="problem-modal-backdrop" {...useBackdropDismiss(() => navigate(-1))} role="presentation">
      <div className="problem-modal" role="dialog" aria-modal="true">
        <button type="button" className="problem-modal-close" onClick={() => navigate(-1)} aria-label="Close">×</button>
        {children}
      </div>
    </div>
  );
}

const VERDICT_MARK: Record<Verdict, string> = { correct: '✓', flag: '⚑', incorrect: '✗' };
const VERDICT_TEXT: Record<Verdict, string> = {
  correct: 'Marked correct', incorrect: 'Marked incorrect', flag: 'Flagged for discussion',
};

/** Pops a problem out over the current view. "Solution" shows the submitted
 * moves; "Explore" is a free analysis board for trying variations together.
 * Plus the teacher's verdict and the original scan. Closes on backdrop/×/Esc. */
export type HistoryEntry = { id: string; moves: Move[]; verdict: Verdict | null; reviewedAt: number; createdAt: number };

export function SolutionModal({
  problem, moves, verdict, reviewedAt, retryHref, defaultMode = 'solution', nav, history, historyOpen = false, onClose,
}: {
  problem: LibProblem;
  moves: Move[];
  verdict?: Verdict | null;
  reviewedAt?: number | null;
  retryHref?: string | null;
  defaultMode?: 'solution' | 'explore';
  nav?: { index: number; total: number; onPrev: () => void; onNext: () => void };
  history?: HistoryEntry[];
  /** Render the attempt history expanded (teacher views); still collapsible. */
  historyOpen?: boolean;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<'solution' | 'explore'>(defaultMode);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev; };
  }, [onClose]);

  useEffect(() => {
    const u = imageUrl(problem.image);
    if (u) u.then(setOriginalSrc);
  }, [problem.image]);

  const stones = toStones(problem.stones);
  const pts = moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(pts);
  const viewport = boundingViewport([...stones, ...pts.map((p) => ({ x: p.x, y: p.y, color: 'B' as const }))], 5);

  return (
    <div className="problem-modal-backdrop" {...useBackdropDismiss(onClose)} role="presentation">
      <div className="problem-modal" role="dialog" aria-modal="true">
        <button type="button" className="problem-modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="solution-modal-header">
          <h2>{problem.collection} <span className="solution-modal-counter">#{problem.source_board_idx + 1}</span></h2>
          <span className="solution-modal-counter">{problem.black_to_play ? 'Black' : 'White'} to play</span>
        </div>

        {nav && nav.total > 1 && (
          <div className="solution-modal-nav">
            <button type="button" onClick={nav.onPrev} disabled={nav.index <= 0}>‹ Prev</button>
            <span className="solution-modal-counter">{nav.index + 1} / {nav.total}</span>
            <button type="button" onClick={nav.onNext} disabled={nav.index >= nav.total - 1}>Next ›</button>
          </div>
        )}

        {verdict && (
          <div className={`solve-verdict-banner v-${verdict}`}>
            <span className="solve-verdict-mark">{VERDICT_MARK[verdict]}</span>
            {VERDICT_TEXT[verdict]}{reviewedAt ? ` · ${new Date(reviewedAt).toLocaleDateString()}` : ''}
          </div>
        )}

        <div className="solution-mode-toggle">
          <button className={mode === 'solution' ? 'active' : ''} onClick={() => setMode('solution')}>Solution</button>
          <button className={mode === 'explore' ? 'active' : ''} onClick={() => setMode('explore')}>Explore</button>
        </div>

        {mode === 'solution' ? (
          <>
            <div className="solution-modal-workspace">
              <div className="solution-modal-board">
                <Board stones={stones} numberedMoves={overlay.boardNumbers} viewport={viewport} displayOnly />
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
            <div className="solution-modal-meta">
              <span>{moves.length} move{moves.length === 1 ? '' : 's'}</span>
              {retryHref && <Link to={retryHref} className="solution-modal-retry" onClick={onClose}>Retry this problem ›</Link>}
            </div>
          </>
        ) : (
          <PlayView key={problem.id} initialStones={stones} />
        )}

        {originalSrc && (
          <details className="solution-modal-original">
            <summary>View original</summary>
            <img src={originalSrc} alt={`Original crop for problem #${problem.source_board_idx + 1}`} />
          </details>
        )}

        {history && history.length > 0 && (
          <details className="solve-history" open={historyOpen || undefined}>
            <summary>Attempt history ({history.length})</summary>
            <ul className="problem-card-grid lg">
              {history.map((h) => (
                <li key={h.id}>
                  <ProblemCard
                    stones={stones}
                    moves={h.moves.map((m) => ({ x: m.col, y: m.row }))}
                    collection={problem.collection}
                    number={problem.source_board_idx + 1}
                    verdict={h.verdict}
                  />
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}
