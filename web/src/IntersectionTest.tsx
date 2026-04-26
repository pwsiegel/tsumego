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
        const r = await api.pdf.detectIntersections(item.page_idx, item.bbox_idx, thresh);
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
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="ix-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {result.intersections.map((p, i) => (
          <circle
            key={i}
            cx={p.x} cy={p.y} r={r}
            fill="rgb(40,180,80)"
            stroke="rgb(20,90,40)"
            strokeWidth={r * 0.3}
            opacity={0.8}
          />
        ))}
      </svg>
    );
  };

  const current = boards[safeSelected];
  const nDet = result?.intersections.length ?? 0;

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
        <div className="ix-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${safeSelected + 1} of ${boards.length}${loading ? ' (detecting…)' : ''}  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}${
                  result ? `  ·  ${nDet} intersections` : ''
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
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx}`}
              className="ix-img"
            />
            {renderOverlay()}
          </div>
        </div>
      )}
    </div>
  );
}
