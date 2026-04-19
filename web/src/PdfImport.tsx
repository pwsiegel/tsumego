import { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';
import type { Stone } from './types';
import './PdfImport.css';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

type Point = { x: number; y: number };

type Props = {
  onImport: (stones: Stone[]) => void;
  onCancel: () => void;
};

const RENDER_SCALE = 2;
const HANDLE_R = 9;

// Preset anchors for where the detected grid maps onto the 19x19.
type AnchorPreset =
  | 'full'
  | 'tl' | 'tr' | 'bl' | 'br'
  | 'top' | 'bottom' | 'left' | 'right'
  | 'custom';

function presetToAnchor(preset: AnchorPreset, rows: number, cols: number): { ax: number; ay: number } {
  switch (preset) {
    case 'full': return { ax: 0, ay: 0 };
    case 'tl':   return { ax: 0, ay: 0 };
    case 'tr':   return { ax: 19 - cols, ay: 0 };
    case 'bl':   return { ax: 0, ay: 19 - rows };
    case 'br':   return { ax: 19 - cols, ay: 19 - rows };
    case 'top':  return { ax: Math.floor((19 - cols) / 2), ay: 0 };
    case 'bottom': return { ax: Math.floor((19 - cols) / 2), ay: 19 - rows };
    case 'left':   return { ax: 0, ay: Math.floor((19 - rows) / 2) };
    case 'right':  return { ax: 19 - cols, ay: Math.floor((19 - rows) / 2) };
    default: return { ax: 0, ay: 0 };
  }
}

type DetectedBox = { x0: number; y0: number; x1: number; y1: number; h_lines: number; v_lines: number };

type LabelBox = { x0: number; y0: number; x1: number; y1: number };

const MIN_LABEL_SIZE = 10;

// Ask the backend to find Go boards in a canvas. Returns an array of bboxes,
// largest first, or [] on failure. Backend uses OpenCV Hough line detection.
async function detectBoardsFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedBox[]> {
  const blob: Blob | null = await new Promise((resolve) =>
    canvas.toBlob((b) => resolve(b), 'image/png')
  );
  if (!blob) return [];
  const form = new FormData();
  form.append('file', blob, 'page.png');
  try {
    const r = await fetch('/api/pdf/detect-boards', { method: 'POST', body: form });
    if (!r.ok) return [];
    const body = await r.json();
    return body.boards as DetectedBox[];
  } catch {
    return [];
  }
}

function bboxToCorners(b: DetectedBox): [Point, Point, Point, Point] {
  return [
    { x: b.x0, y: b.y0 },
    { x: b.x1, y: b.y0 },
    { x: b.x1, y: b.y1 },
    { x: b.x0, y: b.y1 },
  ];
}

export function PdfImport({ onImport, onCancel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const [pageSize, setPageSize] = useState<{ w: number; h: number } | null>(null);
  const [corners, setCorners] = useState<[Point, Point, Point, Point] | null>(null);
  const [gridRows, setGridRows] = useState(19);
  const [gridCols, setGridCols] = useState(19);
  const [preset, setPreset] = useState<AnchorPreset>('full');
  const [anchorX, setAnchorX] = useState(0);
  const [anchorY, setAnchorY] = useState(0);
  const [detected, setDetected] = useState<Array<{ x: number; y: number; color: 'B' | 'W' }> | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pageNum, setPageNum] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [candidates, setCandidates] = useState<DetectedBox[]>([]);
  const [candidateIdx, setCandidateIdx] = useState(0);
  const [detecting, setDetecting] = useState(false);

  const [labelMode, setLabelMode] = useState(false);
  const [labelBoxes, setLabelBoxes] = useState<LabelBox[]>([]);
  const [drawing, setDrawing] = useState<LabelBox | null>(null);
  const [selectedLabelIdx, setSelectedLabelIdx] = useState<number | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [labelCount, setLabelCount] = useState<number | null>(null);

  // Sync preset → anchor.
  useEffect(() => {
    if (preset === 'custom') return;
    const { ax, ay } = presetToAnchor(preset, gridRows, gridCols);
    setAnchorX(ax);
    setAnchorY(ay);
  }, [preset, gridRows, gridCols]);

  const fallbackCorners = (w: number, h: number): [Point, Point, Point, Point] => {
    const m = 0.1;
    return [
      { x: w * m, y: h * m },
      { x: w * (1 - m), y: h * m },
      { x: w * (1 - m), y: h * (1 - m) },
      { x: w * m, y: h * (1 - m) },
    ];
  };

  const renderPage = async (n: number) => {
    const pdf = pdfRef.current;
    if (!pdf) return;
    const page = await pdf.getPage(n);
    const viewport = page.getViewport({ scale: RENDER_SCALE });
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext('2d')!;
    await page.render({ canvas, canvasContext: ctx, viewport }).promise;
    setPageSize({ w: viewport.width, h: viewport.height });
    setDetected(null);
    setCandidates([]);
    setCandidateIdx(0);
    setLabelBoxes([]);
    setSelectedLabelIdx(null);
    setDrawing(null);
    setSaveStatus(null);

    if (labelMode) {
      setCorners(null);
      return;
    }
    setDetecting(true);
    const boards = await detectBoardsFromCanvas(canvas);
    setDetecting(false);
    setCandidates(boards);
    if (boards.length > 0) {
      setCorners(bboxToCorners(boards[0]));
    } else {
      setCorners(fallbackCorners(viewport.width, viewport.height));
    }
  };

  const handleFile = async (file: File) => {
    setError(null);
    setDetected(null);
    setCorners(null);
    try {
      const buf = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
      pdfRef.current = pdf;
      setNumPages(pdf.numPages);
      setPageNum(1);
      await renderPage(1);
    } catch (e) {
      setError(String(e));
    }
  };

  const goToPage = async (n: number) => {
    if (!pdfRef.current) return;
    const clamped = Math.min(Math.max(1, n), numPages);
    setPageNum(clamped);
    await renderPage(clamped);
  };

  const saveLabels = async () => {
    if (!canvasRef.current || labelBoxes.length === 0) return;
    setSaving(true);
    setSaveStatus(null);
    try {
      const canvas = canvasRef.current;
      const blob: Blob | null = await new Promise((res) =>
        canvas.toBlob((b) => res(b), 'image/png')
      );
      if (!blob) throw new Error('canvas toBlob failed');
      const form = new FormData();
      form.append('file', blob, 'page.png');
      form.append(
        'bboxes',
        JSON.stringify(labelBoxes.map((b) => [
          Math.round(b.x0), Math.round(b.y0),
          Math.round(b.x1), Math.round(b.y1),
        ]))
      );
      const r = await fetch('/api/training/save-board-labels', {
        method: 'POST',
        body: form,
      });
      if (!r.ok) throw new Error(`save failed: ${r.status}`);
      const data = await r.json();
      setLabelCount(data.total_labels);
      setSaveStatus(
        `Saved ${data.bbox_count} boxes from page ${pageNum}. ${data.total_labels} total labels.`
      );
      setLabelBoxes([]);
      setSelectedLabelIdx(null);
      if (pageNum < numPages) {
        await goToPage(pageNum + 1);
      }
    } catch (e) {
      setSaveStatus(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  // Keyboard shortcuts for labeling mode.
  useEffect(() => {
    if (!labelMode) return;
    const handler = (e: KeyboardEvent) => {
      const isMetaS = (e.key === 's' || e.key === 'S') && (e.metaKey || e.ctrlKey);
      if (isMetaS) {
        e.preventDefault();
        saveLabels();
        return;
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedLabelIdx !== null) {
        e.preventDefault();
        setLabelBoxes((prev) => prev.filter((_, i) => i !== selectedLabelIdx));
        setSelectedLabelIdx(null);
      }
      if (e.key === 'Escape') {
        setSelectedLabelIdx(null);
        setDrawing(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labelMode, selectedLabelIdx, labelBoxes, pageNum, numPages]);

  const svgPoint = (ev: { clientX: number; clientY: number }): Point => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const ctm = svg.getScreenCTM()!.inverse();
    const p = pt.matrixTransform(ctm);
    return { x: p.x, y: p.y };
  };

  const onHandleDown = (i: number) => (ev: React.PointerEvent) => {
    (ev.target as Element).setPointerCapture(ev.pointerId);
    setDragging(i);
  };
  const onHandleMove = (ev: React.PointerEvent) => {
    if (dragging === null || !corners) return;
    const p = svgPoint(ev);
    const next = corners.slice() as [Point, Point, Point, Point];
    next[dragging] = p;
    setCorners(next);
  };
  const onHandleUp = () => setDragging(null);

  const detect = () => {
    if (!corners || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const [tl, tr, br, bl] = corners;

    const stones: Array<{ x: number; y: number; color: 'B' | 'W' }> = [];

    const approxCellW = (Math.hypot(tr.x - tl.x, tr.y - tl.y) + Math.hypot(br.x - bl.x, br.y - bl.y)) / 2 / Math.max(1, gridCols - 1);
    const approxCellH = (Math.hypot(bl.x - tl.x, bl.y - tl.y) + Math.hypot(br.x - tr.x, br.y - tr.y)) / 2 / Math.max(1, gridRows - 1);
    const cellSize = Math.min(approxCellW, approxCellH);

    const rCenter = Math.max(2, Math.floor(cellSize * 0.12));
    const rInner  = Math.max(3, Math.floor(cellSize * 0.32));
    const rOuter  = Math.max(5, Math.floor(cellSize * 0.48));

    const lerp = (a: Point, b: Point, t: number): Point => ({
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
    });

    // Divide the ring into angular sectors. Empty intersections have dark
    // pixels only where grid lines cross the ring (N/S/E/W → 4 sectors).
    // A stone's outline or fill covers the full ring → most sectors dark.
    const NUM_SECTORS = 12;
    const DARK_LUM = 150;

    const sample = (cx: number, cy: number) => {
      let cSum = 0, cN = 0;
      const sectorDark = new Array(NUM_SECTORS).fill(false);
      for (let dy = -rOuter; dy <= rOuter; dy++) {
        for (let dx = -rOuter; dx <= rOuter; dx++) {
          const d2 = dx * dx + dy * dy;
          if (d2 > rOuter * rOuter) continue;
          const x = Math.round(cx + dx);
          const y = Math.round(cy + dy);
          if (x < 0 || y < 0 || x >= img.width || y >= img.height) continue;
          const i = (y * img.width + x) * 4;
          const lum = 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
          if (d2 <= rCenter * rCenter) {
            cSum += lum;
            cN++;
          }
          if (d2 >= rInner * rInner && lum < DARK_LUM) {
            const angle = Math.atan2(dy, dx) + Math.PI; // 0..2π
            const sectorIdx = Math.min(NUM_SECTORS - 1, Math.floor((angle / (2 * Math.PI)) * NUM_SECTORS));
            sectorDark[sectorIdx] = true;
          }
        }
      }
      let sectors = 0;
      for (const s of sectorDark) if (s) sectors++;
      return {
        centerLum: cN ? cSum / cN : 128,
        sectors,
      };
    };

    // Empty: ~4 sectors (N/S/E/W grid lines). Stones: ~all sectors.
    const MIN_SECTORS_FOR_STONE = 8;
    const CENTER_DARK_THRESH = 110;

    for (let j = 0; j < gridRows; j++) {
      for (let i = 0; i < gridCols; i++) {
        const u = gridCols === 1 ? 0 : i / (gridCols - 1);
        const v = gridRows === 1 ? 0 : j / (gridRows - 1);
        const top = lerp(tl, tr, u);
        const bot = lerp(bl, br, u);
        const p = lerp(top, bot, v);
        const { centerLum, sectors } = sample(p.x, p.y);
        if (sectors >= MIN_SECTORS_FOR_STONE) {
          stones.push({
            x: i,
            y: j,
            color: centerLum < CENTER_DARK_THRESH ? 'B' : 'W',
          });
        }
      }
    }
    setDetected(stones);
  };

  const apply = () => {
    if (!detected) return;
    const out: Stone[] = detected.map((s) => ({
      x: anchorX + s.x,
      y: anchorY + s.y,
      color: s.color,
    })).filter((s) => s.x >= 0 && s.x < 19 && s.y >= 0 && s.y < 19);
    onImport(out);
  };

  // Cycle an intersection's classification: Empty → B → W → Empty.
  const cycleAt = (i: number, j: number) => {
    const current = detected ?? [];
    const idx = current.findIndex((s) => s.x === i && s.y === j);
    if (idx === -1) {
      setDetected([...current, { x: i, y: j, color: 'B' }]);
      return;
    }
    const s = current[idx];
    if (s.color === 'B') {
      const next = current.slice();
      next[idx] = { ...s, color: 'W' };
      setDetected(next);
    } else {
      setDetected(current.filter((_, k) => k !== idx));
    }
  };

  const reset = async () => {
    setDetected(null);
    if (!canvasRef.current || !pageSize) return;
    setDetecting(true);
    const boards = await detectBoardsFromCanvas(canvasRef.current);
    setDetecting(false);
    setCandidates(boards);
    setCandidateIdx(0);
    if (boards.length > 0) {
      setCorners(bboxToCorners(boards[0]));
    } else {
      setCorners(fallbackCorners(pageSize.w, pageSize.h));
    }
  };

  const pickCandidate = (idx: number) => {
    if (idx < 0 || idx >= candidates.length) return;
    setCandidateIdx(idx);
    setCorners(bboxToCorners(candidates[idx]));
    setDetected(null);
  };

  return (
    <div className="pdf-import">
      <div className="pdf-import-toolbar">
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        {numPages > 0 && (
          <div className="pdf-paginator">
            <button
              onClick={() => goToPage(pageNum - 1)}
              disabled={pageNum <= 1}
              aria-label="Previous page"
            >
              ←
            </button>
            <span>
              Page{' '}
              <input
                type="number"
                min={1}
                max={numPages}
                value={pageNum}
                onChange={(e) => goToPage(Number(e.target.value))}
              />{' '}
              / {numPages}
            </span>
            <button
              onClick={() => goToPage(pageNum + 1)}
              disabled={pageNum >= numPages}
              aria-label="Next page"
            >
              →
            </button>
          </div>
        )}
        <button
          className={labelMode ? 'mode-active' : ''}
          onClick={() => {
            setLabelMode((m) => !m);
            setSaveStatus(null);
            setLabelBoxes([]);
            setSelectedLabelIdx(null);
            setDrawing(null);
          }}
        >
          {labelMode ? 'Exit labeling' : 'Label mode'}
        </button>
        {!labelMode && (
          <>
            <button onClick={reset} disabled={!pageSize || detecting}>
              Re-detect
            </button>
            <button
              onClick={async () => {
                if (!canvasRef.current) return;
                const blob: Blob | null = await new Promise((res) =>
                  canvasRef.current!.toBlob((b) => res(b), 'image/png')
                );
                if (!blob) return;
                const form = new FormData();
                form.append('file', blob, 'page.png');
                const r = await fetch('/api/pdf/detect-boards-debug', { method: 'POST', body: form });
                const imgBlob = await r.blob();
                window.open(URL.createObjectURL(imgBlob), '_blank');
              }}
              disabled={!pageSize}
            >
              Debug
            </button>
            {candidates.length > 1 && (
              <div className="pdf-candidates">
                Board{' '}
                <select
                  value={candidateIdx}
                  onChange={(e) => pickCandidate(Number(e.target.value))}
                >
                  {candidates.map((_, i) => (
                    <option key={i} value={i}>{i + 1}</option>
                  ))}
                </select>{' '}
                of {candidates.length}
              </div>
            )}
          </>
        )}
        <button onClick={onCancel}>Cancel</button>
      </div>

      {error && <div className="pdf-error">{error}</div>}

      <div className="pdf-import-body">
        <div className="pdf-stage">
          <div className="pdf-frame">
            <canvas ref={canvasRef} className="pdf-canvas" />
            {pageSize && labelMode && (
              <svg
                ref={svgRef}
                className="pdf-overlay label-overlay"
                viewBox={`0 0 ${pageSize.w} ${pageSize.h}`}
                preserveAspectRatio="none"
                onClick={(ev) => {
                  const p = svgPoint(ev);
                  if (drawing) {
                    const x0 = Math.min(drawing.x0, p.x);
                    const x1 = Math.max(drawing.x0, p.x);
                    const y0 = Math.min(drawing.y0, p.y);
                    const y1 = Math.max(drawing.y0, p.y);
                    if (x1 - x0 >= MIN_LABEL_SIZE && y1 - y0 >= MIN_LABEL_SIZE) {
                      setLabelBoxes((prev) => [...prev, { x0, y0, x1, y1 }]);
                    }
                    setDrawing(null);
                    return;
                  }
                  if (ev.target === ev.currentTarget) {
                    setDrawing({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
                    setSelectedLabelIdx(null);
                  }
                }}
                onPointerMove={(ev) => {
                  if (!drawing) return;
                  const p = svgPoint(ev);
                  setDrawing({ ...drawing, x1: p.x, y1: p.y });
                }}
              >
                {labelBoxes.map((b, i) => (
                  <g key={i}>
                    <rect
                      x={b.x0} y={b.y0}
                      width={b.x1 - b.x0} height={b.y1 - b.y0}
                      className={
                        i === selectedLabelIdx ? 'label-box selected' : 'label-box'
                      }
                      onClick={(ev) => {
                        // While drawing, let the SVG handler commit the rectangle.
                        // Otherwise, select this box.
                        if (drawing) return;
                        ev.stopPropagation();
                        setSelectedLabelIdx(i);
                      }}
                    />
                    {i === selectedLabelIdx && !drawing && (
                      <g
                        onClick={(ev) => {
                          ev.stopPropagation();
                          setLabelBoxes((prev) => prev.filter((_, j) => j !== i));
                          setSelectedLabelIdx(null);
                        }}
                      >
                        <circle cx={b.x1} cy={b.y0} r={16} className="label-delete" />
                        <text
                          x={b.x1} y={b.y0}
                          textAnchor="middle"
                          dominantBaseline="central"
                          className="label-delete-x"
                        >×</text>
                      </g>
                    )}
                  </g>
                ))}
                {drawing && (
                  <rect
                    x={Math.min(drawing.x0, drawing.x1)}
                    y={Math.min(drawing.y0, drawing.y1)}
                    width={Math.abs(drawing.x1 - drawing.x0)}
                    height={Math.abs(drawing.y1 - drawing.y0)}
                    className="label-box drawing"
                  />
                )}
              </svg>
            )}
            {pageSize && !labelMode && corners && (
            <svg
              ref={svgRef}
              className="pdf-overlay"
              viewBox={`0 0 ${pageSize.w} ${pageSize.h}`}
              preserveAspectRatio="none"
              onPointerMove={onHandleMove}
              onPointerUp={onHandleUp}
            >
              {/* Unselected candidates: thin clickable outlines so you can
                  pick one at a glance instead of cycling through a dropdown. */}
              {candidates.map((c, i) => (
                i === candidateIdx ? null : (
                  <rect
                    key={`cand-${i}`}
                    x={c.x0}
                    y={c.y0}
                    width={c.x1 - c.x0}
                    height={c.y1 - c.y0}
                    className="pdf-candidate"
                    onClick={() => pickCandidate(i)}
                  />
                )
              ))}
              <polygon
                points={corners.map((p) => `${p.x},${p.y}`).join(' ')}
                className="pdf-region"
              />
              {/* Grid preview (thin) */}
              {Array.from({ length: gridRows }, (_, j) => (
                <line
                  key={`gh-${j}`}
                  x1={corners[0].x + ((corners[3].x - corners[0].x) * j) / (gridRows - 1)}
                  y1={corners[0].y + ((corners[3].y - corners[0].y) * j) / (gridRows - 1)}
                  x2={corners[1].x + ((corners[2].x - corners[1].x) * j) / (gridRows - 1)}
                  y2={corners[1].y + ((corners[2].y - corners[1].y) * j) / (gridRows - 1)}
                  className="pdf-grid"
                />
              ))}
              {Array.from({ length: gridCols }, (_, i) => (
                <line
                  key={`gv-${i}`}
                  x1={corners[0].x + ((corners[1].x - corners[0].x) * i) / (gridCols - 1)}
                  y1={corners[0].y + ((corners[1].y - corners[0].y) * i) / (gridCols - 1)}
                  x2={corners[3].x + ((corners[2].x - corners[3].x) * i) / (gridCols - 1)}
                  y2={corners[3].y + ((corners[2].y - corners[3].y) * i) / (gridCols - 1)}
                  className="pdf-grid"
                />
              ))}
              {/* Detected stones preview */}
              {detected?.map((s, idx) => {
                const u = gridCols === 1 ? 0 : s.x / (gridCols - 1);
                const v = gridRows === 1 ? 0 : s.y / (gridRows - 1);
                const top = {
                  x: corners[0].x + (corners[1].x - corners[0].x) * u,
                  y: corners[0].y + (corners[1].y - corners[0].y) * u,
                };
                const bot = {
                  x: corners[3].x + (corners[2].x - corners[3].x) * u,
                  y: corners[3].y + (corners[2].y - corners[3].y) * u,
                };
                const p = { x: top.x + (bot.x - top.x) * v, y: top.y + (bot.y - top.y) * v };
                const r = Math.min(
                  Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y) / (gridCols - 1),
                  Math.hypot(corners[3].x - corners[0].x, corners[3].y - corners[0].y) / (gridRows - 1)
                ) * 0.45;
                return (
                  <circle
                    key={`d-${idx}`}
                    cx={p.x}
                    cy={p.y}
                    r={r}
                    className={s.color === 'B' ? 'detected-black' : 'detected-white'}
                  />
                );
              })}
              {/* Click-to-correct hit targets at each intersection.
                  Cycles Empty → B → W → Empty. Placed above detected stones
                  so clicks on a detected circle also route here, but below
                  the corner handles so dragging corners still works. */}
              {Array.from({ length: gridRows * gridCols }, (_, idx) => {
                const i = idx % gridCols;
                const j = Math.floor(idx / gridCols);
                const u = gridCols === 1 ? 0 : i / (gridCols - 1);
                const v = gridRows === 1 ? 0 : j / (gridRows - 1);
                const top = {
                  x: corners[0].x + (corners[1].x - corners[0].x) * u,
                  y: corners[0].y + (corners[1].y - corners[0].y) * u,
                };
                const bot = {
                  x: corners[3].x + (corners[2].x - corners[3].x) * u,
                  y: corners[3].y + (corners[2].y - corners[3].y) * u,
                };
                const p = { x: top.x + (bot.x - top.x) * v, y: top.y + (bot.y - top.y) * v };
                const hitR = Math.min(
                  Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y) / Math.max(1, gridCols - 1),
                  Math.hypot(corners[3].x - corners[0].x, corners[3].y - corners[0].y) / Math.max(1, gridRows - 1)
                ) * 0.5;
                return (
                  <circle
                    key={`hit-${i}-${j}`}
                    cx={p.x}
                    cy={p.y}
                    r={hitR}
                    className="intersection-hit"
                    onClick={() => cycleAt(i, j)}
                  />
                );
              })}
              {/* Corner handles */}
              {corners.map((p, i) => (
                <circle
                  key={`h-${i}`}
                  cx={p.x}
                  cy={p.y}
                  r={HANDLE_R}
                  className="pdf-handle"
                  onPointerDown={onHandleDown(i)}
                />
              ))}
            </svg>
            )}
          </div>
        </div>

        {labelMode ? (
          <div className="pdf-import-sidebar">
            <h2>Labeling</h2>
            <p className="hint">
              Click once at one corner, move to the other, click again to
              commit. Click a box to select (then × or Delete removes it).
              Esc cancels an in-progress box. Cmd/Ctrl+S saves and advances.
            </p>
            <div className="label-stats">
              <div>On this page: <strong>{labelBoxes.length}</strong></div>
              {labelCount !== null && (
                <div>Total saved: <strong>{labelCount}</strong></div>
              )}
            </div>
            <div className="pdf-import-actions">
              <button
                onClick={saveLabels}
                disabled={!pageSize || labelBoxes.length === 0 || saving}
              >
                {saving ? 'Saving…' : `Save ${labelBoxes.length} label${labelBoxes.length === 1 ? '' : 's'} & next page`}
              </button>
            </div>
            {saveStatus && (
              <div className="hint" style={{ marginTop: 8 }}>{saveStatus}</div>
            )}
          </div>
        ) : (
        <div className="pdf-import-sidebar">
          <h2>Grid size</h2>
          <label>
            Rows{' '}
            <input
              type="number"
              min={2}
              max={19}
              value={gridRows}
              onChange={(e) => setGridRows(Math.min(19, Math.max(2, Number(e.target.value))))}
            />
          </label>
          <label>
            Cols{' '}
            <input
              type="number"
              min={2}
              max={19}
              value={gridCols}
              onChange={(e) => setGridCols(Math.min(19, Math.max(2, Number(e.target.value))))}
            />
          </label>

          <h2>Anchor on 19×19</h2>
          <div className="preset-grid">
            {(['tl','top','tr','left','full','right','bl','bottom','br'] as AnchorPreset[]).map((p) => (
              <button
                key={p}
                className={preset === p ? 'active' : ''}
                onClick={() => setPreset(p)}
              >
                {p}
              </button>
            ))}
          </div>
          <label>
            Offset X{' '}
            <input
              type="number"
              min={0}
              max={19 - gridCols}
              value={anchorX}
              onChange={(e) => {
                setPreset('custom');
                setAnchorX(Number(e.target.value));
              }}
            />
          </label>
          <label>
            Offset Y{' '}
            <input
              type="number"
              min={0}
              max={19 - gridRows}
              value={anchorY}
              onChange={(e) => {
                setPreset('custom');
                setAnchorY(Number(e.target.value));
              }}
            />
          </label>

          <div className="pdf-import-actions">
            <button onClick={detect} disabled={!corners}>
              Detect stones
            </button>
            <button onClick={apply} disabled={!detected || detected.length === 0}>
              Apply to board ({detected?.length ?? 0})
            </button>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}
