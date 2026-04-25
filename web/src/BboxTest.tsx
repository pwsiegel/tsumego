import { useEffect, useRef, useState } from 'react';
import { api, type BboxDetectResponse } from './api';
import './BboxTest.css';

type Props = {
  onExit: () => void;
};

export function BboxTest({ onExit }: Props) {
  const [pageCount, setPageCount] = useState<number>(0);
  const [pageIdx, setPageIdx] = useState<number>(0);
  const [detect, setDetect] = useState<BboxDetectResponse | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const inFlight = useRef<number | null>(null);

  const loadPageImage = (idx: number) => {
    setImgSize(null);
    const img = new Image();
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = api.pdf.pageImageUrl(idx);
  };

  const runDetect = async (idx: number) => {
    inFlight.current = idx;
    setLoading(true);
    try {
      const r = await api.pdf.detectBboxes(idx);
      if (inFlight.current !== idx) return;
      setDetect(r);
    } catch (e) {
      if (inFlight.current === idx) {
        setDetect(null);
        setStatus(`Detect failed: ${e}`);
      }
    } finally {
      if (inFlight.current === idx) setLoading(false);
    }
  };

  useEffect(() => {
    if (pageCount === 0) return;
    // loadPageImage owns its own setState; rule can't see through the helper.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadPageImage(pageIdx);
    runDetect(pageIdx);
  }, [pageIdx, pageCount]);

  const uploadPdf = async (file: File) => {
    setUploading(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const data = await api.pdf.uploadPdf(file);
      setPageCount(data.page_count);
      setPageIdx(0);
      setStatus(`${file.name}: ${data.page_count} pages rendered.`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (pageCount === 0) return;
      if (e.key === 'ArrowLeft') setPageIdx((i) => Math.max(0, i - 1));
      if (e.key === 'ArrowRight') setPageIdx((i) => Math.min(pageCount - 1, i + 1));
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [pageCount]);

  const renderOverlay = () => {
    if (!imgSize || !detect) return null;
    return (
      <svg
        viewBox={`0 0 ${detect.page_width} ${detect.page_height}`}
        className="bbox-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {detect.boards.map((b, i) => (
          <g key={i}>
            <rect
              x={b.x0} y={b.y0}
              width={b.x1 - b.x0} height={b.y1 - b.y0}
              fill="none" stroke="rgb(255,60,60)" strokeWidth={Math.max(2, detect.page_width / 400)}
            />
            <text
              x={b.x0 + 4} y={b.y0 + Math.max(14, detect.page_height / 80)}
              fontSize={Math.max(14, detect.page_height / 60)}
              fill="rgb(255,60,60)" fontWeight="bold"
            >
              {i + 1}
            </text>
          </g>
        ))}
      </svg>
    );
  };

  return (
    <div className="bbox-test">
      <div className="bbox-toolbar">
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
        <div className="bbox-status">
          {pageCount > 0
            ? `Page ${pageIdx + 1} of ${pageCount}${loading ? ' (detecting…)' : ''}${detect ? `  ·  ${detect.boards.length} boards` : ''}`
            : 'No PDF uploaded.'}
        </div>
        <div className="bbox-actions">
          <button onClick={() => setPageIdx((i) => Math.max(0, i - 1))}
                  disabled={pageCount === 0 || pageIdx === 0}>◀</button>
          <input
            type="number"
            min={1}
            max={Math.max(1, pageCount)}
            value={pageIdx + 1}
            disabled={pageCount === 0}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isFinite(n)) return;
              setPageIdx(Math.max(0, Math.min(pageCount - 1, n - 1)));
            }}
            style={{ width: '4em' }}
          />
          <button onClick={() => setPageIdx((i) => Math.min(pageCount - 1, i + 1))}
                  disabled={pageCount === 0 || pageIdx >= pageCount - 1}>▶</button>
          <button onClick={onExit}>Done</button>
        </div>
      </div>

      {status && <div className="bbox-message">{status}</div>}

      {pageCount > 0 && (
        <div className="bbox-stage">
          <div className="bbox-panel">
            <img
              src={api.pdf.pageImageUrl(pageIdx)}
              alt={`page ${pageIdx + 1}`}
              className="bbox-img"
            />
            {renderOverlay()}
          </div>
        </div>
      )}
    </div>
  );
}
