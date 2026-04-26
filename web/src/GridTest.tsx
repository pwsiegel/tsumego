import { useEffect, useRef, useState } from 'react';
import { api, type BoardGridDetect, type BoardListItem } from './api';
import './GridTest.css';

type Props = {
  onExit: () => void;
};

export function GridTest({ onExit }: Props) {
  const [boards, setBoards] = useState<BoardListItem[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [result, setResult] = useState<BoardGridDetect | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
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

  const loadCurrent = (item: BoardListItem) => {
    setResult(null);
    const key = `${item.page_idx}:${item.bbox_idx}`;
    inFlight.current = key;
    setLoading(true);
    (async () => {
      try {
        const r = await api.pdf.detectGrid(item.page_idx, item.bbox_idx);
        if (inFlight.current !== key) return;
        setResult(r);
      } catch (e) {
        if (inFlight.current === key) setStatus(`Grid detect failed: ${e}`);
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
    loadCurrent(boards[safeSelected]);
  }, [boards, safeSelected]);

  const uploadPdf = async (file: File) => {
    setUploading(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const data = await api.pdf.uploadPdf(file);
      setStatus(`${file.name}: ${data.page_count} pages rendered. Running YOLO…`);
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
    const { grid_x0, grid_y0, grid_x1, grid_y1, pitch_x, pitch_y, edges } = result;
    const stroke = Math.max(1, W / 600);

    // Grid lines: from grid_x0 stepping by pitch_x until past grid_x1.
    // Use a small epsilon so the right/bottom edges aren't dropped to FP error.
    const eps = 1e-3;
    const vLines: number[] = [];
    for (let x = grid_x0; x <= grid_x1 + eps; x += pitch_x) vLines.push(x);
    const hLines: number[] = [];
    for (let y = grid_y0; y <= grid_y1 + eps; y += pitch_y) hLines.push(y);

    const edgeColor = (on: boolean) => (on ? 'rgb(40,180,80)' : 'rgb(180,180,180)');
    const badgePad = Math.max(6, W / 80);

    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="grid-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {/* Predicted grid bbox */}
        <rect
          x={grid_x0} y={grid_y0}
          width={grid_x1 - grid_x0} height={grid_y1 - grid_y0}
          fill="rgba(80,160,220,0.08)"
          stroke="rgb(50,140,210)"
          strokeWidth={stroke * 2}
          strokeDasharray={`${stroke * 6} ${stroke * 4}`}
        />
        {/* Vertical grid lines */}
        {vLines.map((x, i) => (
          <line
            key={`v${i}`}
            x1={x} x2={x} y1={grid_y0} y2={grid_y1}
            stroke="rgb(220,80,80)"
            strokeWidth={stroke}
            opacity={0.7}
          />
        ))}
        {/* Horizontal grid lines */}
        {hLines.map((y, i) => (
          <line
            key={`h${i}`}
            x1={grid_x0} x2={grid_x1} y1={y} y2={y}
            stroke="rgb(220,80,80)"
            strokeWidth={stroke}
            opacity={0.7}
          />
        ))}
        {/* Edge indicators (small bars hugging each crop side) */}
        <line
          x1={badgePad} x2={badgePad} y1={H * 0.2} y2={H * 0.8}
          stroke={edgeColor(edges.left)} strokeWidth={stroke * 5}
        />
        <line
          x1={W - badgePad} x2={W - badgePad} y1={H * 0.2} y2={H * 0.8}
          stroke={edgeColor(edges.right)} strokeWidth={stroke * 5}
        />
        <line
          x1={W * 0.2} x2={W * 0.8} y1={badgePad} y2={badgePad}
          stroke={edgeColor(edges.top)} strokeWidth={stroke * 5}
        />
        <line
          x1={W * 0.2} x2={W * 0.8} y1={H - badgePad} y2={H - badgePad}
          stroke={edgeColor(edges.bottom)} strokeWidth={stroke * 5}
        />
      </svg>
    );
  };

  const current = boards[safeSelected];
  const edgeTxt = result
    ? (['left', 'right', 'top', 'bottom'] as const)
        .filter((k) => result.edges[k]).join('+') || 'none'
    : '—';
  const linesText = result
    ? `${Math.round((result.grid_x1 - result.grid_x0) / result.pitch_x) + 1}` +
      `×${Math.round((result.grid_y1 - result.grid_y0) / result.pitch_y) + 1} lines`
    : '';

  return (
    <div className="grid-test">
      <div className="grid-toolbar">
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
        <div className="grid-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${safeSelected + 1} of ${boards.length}${loading ? ' (detecting…)' : ''}` +
                `  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}` +
                (result
                  ? `  ·  pitch ${result.pitch_x.toFixed(1)}×${result.pitch_y.toFixed(1)} px` +
                    `  ·  ${linesText}  ·  edges: ${edgeTxt}`
                  : '')
              : ''}
        </div>
        <div className="grid-actions">
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

      {status && <div className="grid-message">{status}</div>}

      {current && (
        <div className="grid-stage">
          <div className="grid-panel">
            <img
              src={api.pdf.boardCropUrl(current.page_idx, current.bbox_idx)}
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx}`}
              className="grid-img"
            />
            {renderOverlay()}
          </div>
        </div>
      )}
    </div>
  );
}
