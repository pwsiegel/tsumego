import { useEffect, useRef, useState } from 'react';
import './StoneTest.css';

type BoardListItem = {
  page_idx: number;
  bbox_idx: number;
  x0: number; y0: number; x1: number; y1: number;
  confidence: number;
};

type CnnStone = {
  x: number;
  y: number;
  r: number;
  color: string;   // "B" or "W"
  conf: number;
};

type BoardStonesLocal = {
  page_idx: number;
  bbox_idx: number;
  crop_width: number;
  crop_height: number;
  stones: CnnStone[];
};

type Props = {
  onExit: () => void;
};

export function StoneTest({ onExit }: Props) {
  const [boards, setBoards] = useState<BoardListItem[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [result, setResult] = useState<BoardStonesLocal | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [peakThresh, setPeakThresh] = useState<number>(0.3);
  const [status, setStatus] = useState<string | null>(null);
  const inFlight = useRef<string | null>(null);

  const refreshBoards = async () => {
    const r = await fetch('/api/pdf/boards', { cache: 'no-store' });
    if (!r.ok) {
      setBoards([]);
      return;
    }
    const data = await r.json();
    setBoards(data.boards);
    setSelected(0);
  };

  useEffect(() => {
    refreshBoards();
  }, []);

  const loadCurrent = (item: BoardListItem, thresh: number) => {
    setImgSize(null);
    setResult(null);
    const img = new Image();
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = `/api/pdf/board-crop/${item.page_idx}/${item.bbox_idx}.png?_t=${Date.now()}`;

    const key = `${item.page_idx}:${item.bbox_idx}:${thresh}`;
    inFlight.current = key;
    setLoading(true);
    (async () => {
      try {
        const r = await fetch(
          `/api/pdf/board-stones/${item.page_idx}/${item.bbox_idx}?peak_thresh=${thresh}&_t=${Date.now()}`,
          { cache: 'no-store' },
        );
        if (inFlight.current !== key) return;
        if (r.ok) setResult((await r.json()) as BoardStonesLocal);
        else {
          const body = await r.json().catch(() => ({ detail: r.statusText }));
          setStatus(`Stone detect failed: ${body.detail ?? r.statusText}`);
        }
      } finally {
        if (inFlight.current === key) setLoading(false);
      }
    })();
  };

  useEffect(() => {
    if (boards.length === 0) return;
    const clamped = Math.max(0, Math.min(boards.length - 1, selected));
    if (clamped !== selected) {
      setSelected(clamped);
      return;
    }
    loadCurrent(boards[clamped], peakThresh);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boards, selected, peakThresh]);

  const uploadPdf = async (file: File) => {
    setUploading(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      const r = await fetch('/api/pdf/bbox-upload', { method: 'POST', body: form });
      if (!r.ok) {
        const body = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(body.detail ?? r.statusText);
      }
      const data = await r.json();
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
    if (!imgSize || !result) return null;
    const W = result.crop_width;
    const H = result.crop_height;
    const stroke = Math.max(1, W / 400);
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="stone-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {result.stones.map((s, i) => (
          <g key={i}>
            <circle
              cx={s.x} cy={s.y} r={s.r}
              fill="none"
              stroke={s.color === 'B' ? 'rgb(40,180,80)' : 'rgb(220,80,80)'}
              strokeWidth={stroke * 2}
              opacity={0.9}
            />
            <circle
              cx={s.x} cy={s.y} r={2}
              fill={s.color === 'B' ? 'rgb(40,180,80)' : 'rgb(220,80,80)'}
            />
          </g>
        ))}
      </svg>
    );
  };

  const current = boards[selected];
  const bCount = result?.stones.filter((s) => s.color === 'B').length ?? 0;
  const wCount = result?.stones.filter((s) => s.color === 'W').length ?? 0;

  return (
    <div className="stone-test">
      <div className="stone-toolbar">
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
        <div className="stone-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${selected + 1} of ${boards.length}${loading ? ' (detecting…)' : ''}  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}${
                  result ? `  ·  ${bCount} B, ${wCount} W` : ''
                }`
              : ''}
        </div>
        <div className="stone-actions">
          <button onClick={() => setSelected((i) => Math.max(0, i - 1))}
                  disabled={boards.length === 0 || selected === 0}>◀</button>
          <input
            type="number"
            min={1}
            max={Math.max(1, boards.length)}
            value={selected + 1}
            disabled={boards.length === 0}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isFinite(n)) return;
              setSelected(Math.max(0, Math.min(boards.length - 1, n - 1)));
            }}
            style={{ width: '4em' }}
          />
          <button onClick={() => setSelected((i) => Math.min(boards.length - 1, i + 1))}
                  disabled={boards.length === 0 || selected >= boards.length - 1}>▶</button>
          <button onClick={onExit}>Done</button>
        </div>
      </div>

      {status && <div className="stone-message">{status}</div>}

      {current && (
        <div className="stone-stage">
          <div className="stone-panel">
            <img
              src={`/api/pdf/board-crop/${current.page_idx}/${current.bbox_idx}.png?_t=${Date.now()}`}
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx}`}
              className="stone-img"
            />
            {renderOverlay()}
          </div>
        </div>
      )}
    </div>
  );
}
