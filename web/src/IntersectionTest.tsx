import { useEffect, useRef, useState } from 'react';
import { api, type BoardIntersections, type BoardListItem } from './api';
import './IntersectionTest.css';

type Props = {
  onExit: () => void;
};

export function IntersectionTest({ onExit }: Props) {
  const [boards, setBoards] = useState<BoardListItem[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [result, setResult] = useState<BoardIntersections | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [peakThresh, setPeakThresh] = useState<number>(0.3);
  const [showLattice, setShowLattice] = useState<boolean>(false);
  const [showSegments, setShowSegments] = useState<boolean>(true);
  const [status, setStatus] = useState<string | null>(null);
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
    setResult(null);
    const key = `${item.page_idx}:${item.bbox_idx}:${thresh}`;
    inFlight.current = key;
    setLoading(true);
    (async () => {
      try {
        const r = await api.pdf.detectIntersections(
          item.page_idx, item.bbox_idx, thresh,
        );
        if (inFlight.current !== key) return;
        setResult(r);
      } catch (e) {
        if (inFlight.current === key) setStatus(`Detect failed: ${e}`);
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
    setUploading(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const data = await api.pdf.uploadPdf(file);
      setStatus(`${file.name}: ${data.page_count} pages rendered. Running YOLO on all pages…`);
      await refreshBoards();
      setStatus(`${file.name}: ${data.page_count} pages.`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setUploading(false);
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

  const renderOverlay = () => {
    if (!result) return null;
    const W = result.crop_width;
    const H = result.crop_height;
    const r = Math.max(2, Math.min(W, H) / 100);
    const cls = result.fused_lattice;
    const segs = result.segments ?? [];

    // Color LSD segments by orientation: near-horizontal green, near-vertical
    // blue, other gray. Lets the user eyeball whether the segment extractor
    // is picking up grid lines cleanly vs noise.
    const segColor = (s: { x1: number; y1: number; x2: number; y2: number }) => {
      const ang = Math.atan2(s.y2 - s.y1, s.x2 - s.x1) * 180 / Math.PI;
      const a = Math.abs(((ang + 180) % 180));
      if (a < 15 || a > 165) return 'rgb(0,180,80)';
      if (a > 75 && a < 105) return 'rgb(40,80,220)';
      return 'rgb(160,160,160)';
    };
    const skelJ = result.skeleton_junctions ?? [];
    const junctionColor: Record<string, string> = {
      T: 'rgb(220, 60, 160)',  // magenta
      L: 'rgb(255, 140, 0)',   // orange
      '+': 'rgb(80, 200, 120)', // green
      I: 'rgb(160, 160, 160)',
      '?': 'rgb(160, 160, 160)',
    };
    const renderJunctions = () => skelJ.map((j, i) => (
      <circle key={`j${i}`} cx={j.x} cy={j.y} r={r * 0.6}
              fill={junctionColor[j.kind] ?? 'rgb(160,160,160)'}
              opacity={0.9} />
    ));
    if (!cls || !showLattice) {
      return (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="ix-overlay"
          preserveAspectRatio="none"
          style={{ pointerEvents: 'none' }}
        >
          {showSegments && segs.map((s, i) => (
            <line key={`s${i}`} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                  stroke={segColor(s)} strokeWidth={r * 0.3} opacity={0.85} />
          ))}
          {renderJunctions()}
        </svg>
      );
    }

    // Classical grid lines run across the whole crop. Either axis may be
    // unresolved, so we draw what's available independently.
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    if (cls.pitch_x != null && cls.origin_x != null) {
      const px = cls.pitch_x;
      const cMinScan = Math.floor((0 - cls.origin_x) / px) - 1;
      const cMaxScan = Math.ceil((W - cls.origin_x) / px) + 1;
      for (let c = cMinScan; c <= cMaxScan; c++) {
        const x = cls.origin_x + c * px;
        lines.push({ x1: x, y1: 0, x2: x, y2: H });
      }
    }
    if (cls.pitch_y != null && cls.origin_y != null) {
      const py = cls.pitch_y;
      const rMinScan = Math.floor((0 - cls.origin_y) / py) - 1;
      const rMaxScan = Math.ceil((H - cls.origin_y) / py) + 1;
      for (let row = rMinScan; row <= rMaxScan; row++) {
        const y = cls.origin_y + row * py;
        lines.push({ x1: 0, y1: y, x2: W, y2: y });
      }
    }

    const edgeStroke = (real: boolean) => (real ? 'rgb(0,170,200)' : 'rgb(140,200,210)');
    const edgeDash = (real: boolean) => (real ? undefined : `${r * 1.2},${r * 0.8}`);
    const edgeWidth = (real: boolean) => (real ? r * 0.5 : r * 0.25);

    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="ix-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {showSegments && segs.map((s, i) => (
          <line key={`s${i}`} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                stroke={segColor(s)} strokeWidth={r * 0.3} opacity={0.85} />
        ))}

        {lines.map((l, i) => (
          <line key={`g${i}`} {...l}
                stroke="rgb(0,170,200)" strokeWidth={r * 0.15} opacity={0.55} />
        ))}

        {renderJunctions()}

        {/* Edge rectangle: 19-line standard board (18 pitches from origin).
            Solid for "real" edges, dashed lighter cyan for virtual. */}
        {cls.pitch_x != null && cls.pitch_y != null
          && cls.origin_x != null && cls.origin_y != null
          && (() => {
          const left = cls.origin_x;
          const right = cls.origin_x + 18 * cls.pitch_x;
          const top = cls.origin_y;
          const bottom = cls.origin_y + 18 * cls.pitch_y;
          return (
            <g>
              <line x1={left} y1={top} x2={left} y2={bottom}
                    stroke={edgeStroke(cls.edges.left)} strokeWidth={edgeWidth(cls.edges.left)}
                    strokeDasharray={edgeDash(cls.edges.left)} />
              <line x1={right} y1={top} x2={right} y2={bottom}
                    stroke={edgeStroke(cls.edges.right)} strokeWidth={edgeWidth(cls.edges.right)}
                    strokeDasharray={edgeDash(cls.edges.right)} />
              <line x1={left} y1={top} x2={right} y2={top}
                    stroke={edgeStroke(cls.edges.top)} strokeWidth={edgeWidth(cls.edges.top)}
                    strokeDasharray={edgeDash(cls.edges.top)} />
              <line x1={left} y1={bottom} x2={right} y2={bottom}
                    stroke={edgeStroke(cls.edges.bottom)} strokeWidth={edgeWidth(cls.edges.bottom)}
                    strokeDasharray={edgeDash(cls.edges.bottom)} />
            </g>
          );
        })()}
      </svg>
    );
  };

  const current = boards[safeSelected];
  const cls = result?.fused_lattice ?? null;
  const fmt = (v: number | null) => (v == null ? '?' : v.toFixed(1));
  const clsSummary = cls
    ? `pitch ${fmt(cls.pitch_x)},${fmt(cls.pitch_y)}  edges ${
        (['left', 'right', 'top', 'bottom'] as const)
          .filter((s) => cls.edges[s]).join('/') || '∅'
      }`
    : '∅';

  return (
    <div className="ix-test">
      <div className="ix-toolbar">
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
        <label className="peak-slider">
          peak_thresh: {peakThresh.toFixed(2)}
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={peakThresh}
            onChange={(e) => setPeakThresh(Number(e.target.value))}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={showLattice}
            onChange={(e) => setShowLattice(e.target.checked)}
          />
          {' lattice'}
        </label>
        <label>
          <input
            type="checkbox"
            checked={showSegments}
            onChange={(e) => setShowSegments(e.target.checked)}
          />
          {' segments'}
        </label>
        <div className="ix-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${safeSelected + 1} of ${boards.length}${loading ? ' (detecting…)' : ''}  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}${
                  result ? `  ·  ${clsSummary}  ·  ${result.segments?.length ?? 0} segs  ·  ${result.skeleton_junctions?.length ?? 0} jns` : ''
                }`
              : ''}
        </div>
        <div className="ix-actions">
          <button onClick={() => setSelected(Math.max(0, safeSelected - 1))}
                  disabled={boards.length === 0 || safeSelected === 0}>◀</button>
          <input
            type="number"
            min={1}
            max={Math.max(1, boards.length)}
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

      {status && <div className="ix-message">{status}</div>}

      {current && (
        <div className="ix-stage">
          <div className="ix-panel">
            <img
              src={api.pdf.boardCropUrl(current.page_idx, current.bbox_idx)}
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx} (original)`}
              className="ix-img"
            />
          </div>
          <div className="ix-panel">
            <img
              src={api.pdf.boardSkeletonUrl(current.page_idx, current.bbox_idx)}
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx} (skeleton)`}
              className="ix-img"
            />
            {renderOverlay()}
          </div>
        </div>
      )}
    </div>
  );
}
