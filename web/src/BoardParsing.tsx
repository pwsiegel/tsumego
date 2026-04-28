import { useEffect, useRef, useState } from 'react';
import { Board } from './Board';
import {
  api,
  type BoardDiscretize,
  type BoardIntersections,
  type BoardListItem,
  type Junction,
  type Segment,
} from './api';

type IngestProgress = {
  stage: 'render' | 'detect';
  done: number;
  total: number;
};
import type { Stone } from './types';
import './BoardParsing.css';

type Props = { onExit: () => void };

export function BoardParsing({ onExit }: Props) {
  const [boards, setBoards] = useState<BoardListItem[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [ix, setIx] = useState<BoardIntersections | null>(null);
  const [disc, setDisc] = useState<BoardDiscretize | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [peakThresh, setPeakThresh] = useState<number>(0.3);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState<IngestProgress | null>(null);

  // Layer toggles. Stones overlay is on the PDF (left); the other four
  // are independent overlays on the right panel.
  const [showStones, setShowStones] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showSegments, setShowSegments] = useState(true);
  const [showEdges, setShowEdges] = useState(true);
  const [showLattice, setShowLattice] = useState(false);

  const inFlight = useRef<string | null>(null);

  const refreshBoards = async () => {
    try {
      setBoards(await api.pdf.listBoards());
      setSelected(0);
    } catch {
      setBoards([]);
    }
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshBoards();
  }, []);

  const loadCurrent = (item: BoardListItem, thresh: number) => {
    setIx(null);
    setDisc(null);
    const key = `${item.page_idx}:${item.bbox_idx}:${thresh}`;
    inFlight.current = key;
    setLoading(true);
    (async () => {
      try {
        const [ixRes, discRes] = await Promise.all([
          api.pdf.detectIntersections(item.page_idx, item.bbox_idx, thresh),
          api.pdf.discretizeBoard(item.page_idx, item.bbox_idx, thresh),
        ]);
        if (inFlight.current !== key) return;
        setIx(ixRes);
        setDisc(discRes);
      } catch (e) {
        if (inFlight.current === key) setStatus(`Parse failed: ${e}`);
      } finally {
        if (inFlight.current === key) setLoading(false);
      }
    })();
  };

  const safeSelected = boards.length === 0
    ? 0
    : Math.max(0, Math.min(boards.length - 1, selected));

  useEffect(() => {
    if (boards.length === 0) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadCurrent(boards[safeSelected], peakThresh);
  }, [boards, safeSelected, peakThresh]);

  const uploadPdf = async (file: File) => {
    // Clear stale state immediately so we don't show the previous PDF's
    // boards/overlays while the new upload is being processed.
    inFlight.current = null;
    setBoards([]);
    setIx(null);
    setDisc(null);
    setSelected(0);
    setUploading(true);
    setStatus(`Uploading ${file.name}…`);
    setProgress({ stage: 'render', done: 0, total: 0 });

    const accumulated: BoardListItem[] = [];
    try {
      await api.pdf.uploadPdfStream(file, (event) => {
        if (event.event === 'start') {
          setProgress({ stage: event.stage, done: 0, total: event.total });
        } else if (event.event === 'page-rendered') {
          setProgress((p) => p ? { ...p, done: p.done + 1 } : p);
        } else if (event.event === 'page-detected') {
          accumulated.push(...event.boards);
          setProgress((p) => p ? { ...p, done: p.done + 1 } : p);
        } else if (event.event === 'done') {
          setStatus(`${file.name}: ${event.page_count} pages, ${event.board_count} boards.`);
        } else if (event.event === 'error') {
          setStatus(`Error: ${event.message}`);
        }
      });
      setBoards(accumulated);
      setSelected(0);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setUploading(false);
      setProgress(null);
    }
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (boards.length === 0) return;
      if (e.key === 'ArrowLeft') setSelected((i) => Math.max(0, i - 1));
      if (e.key === 'ArrowRight') setSelected((i) => Math.min(boards.length - 1, i + 1));
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [boards.length]);

  const current = boards[safeSelected];
  const W = ix?.crop_width ?? disc?.crop_width ?? 0;
  const H = ix?.crop_height ?? disc?.crop_height ?? 0;

  const sgfStones: Stone[] = (disc?.stones ?? []).map((s) => ({
    x: s.col,
    y: s.row,
    color: s.color === 'B' ? 'B' : 'W',
  }));

  // Crop the rendered board to ~2 cells past the stones on each side.
  // Reveals the local context the player cares about without showing
  // 19x19 of empty space when a corner problem covers a third of the
  // board.
  const PAD = 2;
  const viewport = sgfStones.length > 0 ? {
    colMin: Math.max(0, Math.min(...sgfStones.map((s) => s.x)) - PAD),
    colMax: Math.min(18, Math.max(...sgfStones.map((s) => s.x)) + PAD),
    rowMin: Math.max(0, Math.min(...sgfStones.map((s) => s.y)) - PAD),
    rowMax: Math.min(18, Math.max(...sgfStones.map((s) => s.y)) + PAD),
  } : undefined;

  return (
    <div className="bp-test">
      <div className="bp-toolbar">
        <label className="upload-btn">
          {uploading ? 'Uploading…' : 'Upload PDF'}
          <input
            type="file"
            accept="application/pdf"
            disabled={uploading}
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) uploadPdf(f);
              e.target.value = '';
            }}
          />
        </label>
        <div className="bp-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${safeSelected + 1} of ${boards.length}${loading ? ' (parsing…)' : ''}  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}${
                  disc ? `  ·  edges: ${edgeSummary(disc.edges)}` : ''
                }`
              : ''}
        </div>
        <div className="bp-actions">
          <button onClick={() => setSelected(Math.max(0, safeSelected - 1))}
                  disabled={boards.length === 0 || safeSelected === 0}>◀</button>
          <input
            type="number" min={1} max={Math.max(1, boards.length)}
            value={safeSelected + 1}
            disabled={boards.length === 0}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isFinite(n)) return;
              setSelected(Math.max(0, Math.min(boards.length - 1, n - 1)));
            }}
            style={{ width: '4em' }}
          />
          <button onClick={() => setSelected(Math.min(boards.length - 1, safeSelected + 1))}
                  disabled={boards.length === 0 || safeSelected >= boards.length - 1}>▶</button>
          <button onClick={onExit}>Done</button>
        </div>
      </div>
      <div className="bp-toolbar bp-toolbar-row2">
        <label className="peak-slider">
          peak_thresh: {peakThresh.toFixed(2)}
          <input
            type="range" min={0.05} max={0.95} step={0.05}
            value={peakThresh}
            onChange={(e) => setPeakThresh(Number(e.target.value))}
          />
        </label>
        <ToggleLabel label="stones" checked={showStones} onChange={setShowStones} />
        <ToggleLabel label="skeleton" checked={showSkeleton} onChange={setShowSkeleton} />
        <ToggleLabel label="segments" checked={showSegments} onChange={setShowSegments} />
        <ToggleLabel label="edges" checked={showEdges} onChange={setShowEdges} />
        <ToggleLabel label="lattice" checked={showLattice} onChange={setShowLattice} />
      </div>

      {status && <div className="bp-message">{status}</div>}

      {progress && <IngestProgressBar progress={progress} />}

      {current && W > 0 && H > 0 && (
        <>
          <div className="bp-stage">
            <div className="bp-panel">
              <img
                src={api.pdf.boardCropUrl(current.page_idx, current.bbox_idx)}
                alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx}`}
                className="bp-img"
              />
              {showStones && ix && (
                <svg
                  viewBox={`0 0 ${W} ${H}`}
                  className="bp-overlay"
                  preserveAspectRatio="none"
                  style={{ pointerEvents: 'none' }}
                >
                  {ix.stones.map((s, i) => {
                    // Match the actual stone size (a real stone fills
                    // ~0.45 of the cell) so a misfit pitch is visible
                    // as circles that don't line up with the discs.
                    const r = disc ? Math.max(4, disc.cell_size * 0.45) : Math.max(4, Math.min(W, H) / 60);
                    return (
                      <circle
                        key={i}
                        cx={s.x} cy={s.y} r={r}
                        fill="none"
                        stroke={s.color === 'B' ? 'rgb(40,180,80)' : 'rgb(220,80,80)'}
                        strokeWidth={Math.max(1, r * 0.25)}
                        opacity={0.9}
                      />
                    );
                  })}
                </svg>
              )}
            </div>
            <div className="bp-panel">
              <img
                src={api.pdf.boardSkeletonUrl(current.page_idx, current.bbox_idx)}
                alt={`skeleton ${current.page_idx + 1}/${current.bbox_idx}`}
                className="bp-img"
                style={{ visibility: showSkeleton ? 'visible' : 'hidden' }}
              />
              {ix && (
                <FeatureOverlay
                  W={W} H={H}
                  ix={ix}
                  showSegments={showSegments}
                  showEdges={showEdges}
                  showLattice={showLattice}
                />
              )}
            </div>
          </div>

          <div className="bp-board">
            {disc && (
              <Board stones={sgfStones} onPlay={() => {}} displayOnly viewport={viewport} />
            )}
          </div>
        </>
      )}
    </div>
  );
}

function IngestProgressBar({ progress }: { progress: IngestProgress }) {
  const { stage, done, total } = progress;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const label = stage === 'render' ? 'Rendering pages' : 'Detecting boards';
  return (
    <div className="bp-progress">
      <div className="bp-progress-label">
        {label}: {done} / {total || '?'} ({pct}%)
      </div>
      <div className="bp-progress-track">
        <div className="bp-progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ToggleLabel({
  label, checked, onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="bp-toggle">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {' '}{label}
    </label>
  );
}

function edgeSummary(e: { left: boolean; right: boolean; top: boolean; bottom: boolean }): string {
  const sides = (['left', 'right', 'top', 'bottom'] as const).filter((s) => e[s]);
  return sides.length === 0 ? '∅' : sides.join('/');
}

function segColor(s: Segment): string {
  const ang = Math.atan2(s.y2 - s.y1, s.x2 - s.x1) * 180 / Math.PI;
  const a = Math.abs(((ang + 180) % 180));
  if (a < 15 || a > 165) return 'rgb(0,180,80)';
  if (a > 75 && a < 105) return 'rgb(40,80,220)';
  return 'rgb(160,160,160)';
}

const JUNCTION_COLOR: Record<string, string> = {
  T: 'rgb(220, 60, 160)',
  L: 'rgb(255, 140, 0)',
  '+': 'rgb(80, 200, 120)',
  I: 'rgb(160, 160, 160)',
  '?': 'rgb(160, 160, 160)',
};

function FeatureOverlay({
  W, H, ix, showSegments, showEdges, showLattice,
}: {
  W: number;
  H: number;
  ix: BoardIntersections;
  showSegments: boolean;
  showEdges: boolean;
  showLattice: boolean;
}) {
  const r = Math.max(2, Math.min(W, H) / 100);
  const segs = ix.segments ?? [];
  const junctions = ix.skeleton_junctions ?? [];
  const cls = ix.fused_lattice;

  const latticeLines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  if (showLattice && cls) {
    if (cls.pitch_x != null && cls.origin_x != null) {
      const px = cls.pitch_x;
      const cMin = Math.floor((0 - cls.origin_x) / px) - 1;
      const cMax = Math.ceil((W - cls.origin_x) / px) + 1;
      for (let c = cMin; c <= cMax; c++) {
        const x = cls.origin_x + c * px;
        latticeLines.push({ x1: x, y1: 0, x2: x, y2: H });
      }
    }
    if (cls.pitch_y != null && cls.origin_y != null) {
      const py = cls.pitch_y;
      const rMin = Math.floor((0 - cls.origin_y) / py) - 1;
      const rMax = Math.ceil((H - cls.origin_y) / py) + 1;
      for (let row = rMin; row <= rMax; row++) {
        const y = cls.origin_y + row * py;
        latticeLines.push({ x1: 0, y1: y, x2: W, y2: y });
      }
    }
  }

  const edgeBox = (() => {
    if (!showEdges || !cls) return null;
    if (cls.pitch_x == null || cls.pitch_y == null
        || cls.origin_x == null || cls.origin_y == null) return null;
    const left = cls.origin_x;
    const right = cls.origin_x + 18 * cls.pitch_x;
    const top = cls.origin_y;
    const bottom = cls.origin_y + 18 * cls.pitch_y;
    return { left, right, top, bottom, e: cls.edges };
  })();

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="bp-overlay"
      preserveAspectRatio="none"
      style={{ pointerEvents: 'none' }}
    >
      {showSegments && segs.map((s, i) => (
        <line key={`s${i}`}
              x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
              stroke={segColor(s)} strokeWidth={r * 0.3} opacity={0.85} />
      ))}

      {showLattice && latticeLines.map((l, i) => (
        <line key={`g${i}`} {...l}
              stroke="rgb(0,170,200)" strokeWidth={r * 0.15} opacity={0.55} />
      ))}

      {showEdges && junctions.map((j: Junction, i) => (
        <circle key={`j${i}`}
                cx={j.x} cy={j.y} r={r * 0.6}
                fill={JUNCTION_COLOR[j.kind] ?? 'rgb(160,160,160)'}
                opacity={0.9} />
      ))}

      {edgeBox && (
        <g>
          {edgeBox.e.left && (
            <line x1={edgeBox.left} y1={edgeBox.top} x2={edgeBox.left} y2={edgeBox.bottom}
                  stroke="rgb(0,170,200)" strokeWidth={r * 0.6} />
          )}
          {edgeBox.e.right && (
            <line x1={edgeBox.right} y1={edgeBox.top} x2={edgeBox.right} y2={edgeBox.bottom}
                  stroke="rgb(0,170,200)" strokeWidth={r * 0.6} />
          )}
          {edgeBox.e.top && (
            <line x1={edgeBox.left} y1={edgeBox.top} x2={edgeBox.right} y2={edgeBox.top}
                  stroke="rgb(0,170,200)" strokeWidth={r * 0.6} />
          )}
          {edgeBox.e.bottom && (
            <line x1={edgeBox.left} y1={edgeBox.bottom} x2={edgeBox.right} y2={edgeBox.bottom}
                  stroke="rgb(0,170,200)" strokeWidth={r * 0.6} />
          )}
        </g>
      )}
    </svg>
  );
}
