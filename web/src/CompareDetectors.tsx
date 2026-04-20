import { useEffect, useMemo, useRef, useState } from 'react';
import { Board } from './Board';
import type { Stone } from './types';
import './StoneLabeler.css';
import './CompareDetectors.css';

type Task = {
  task_id: string;
  source: string;
  labeled: boolean;
};

type Circle = {
  x: number;
  y: number;
  r: number;
  color: 'B' | 'W' | null;
  conf?: number;
};

type SgfResult = {
  sgf: string;
  stones: Array<{ col: number; row: number; color: 'B' | 'W' }>;
  pitch: number;
  edges_detected: Record<string, boolean>;
};

type EdgeProbs = { left: number; right: number; top: number; bottom: number };

type GridResult = {
  grid: number[][];   // 19x19 of 0/1/2
  edges: Record<string, boolean>;
  window: { col_min: number; col_max: number; row_min: number; row_max: number };
  sgf: string;
};

type PipelineResult = {
  stones: Array<{ col: number; row: number; color: 'B' | 'W'; x_px: number; y_px: number }>;
  sgf: string;
  edges: Record<string, boolean>;
  window: { col_min: number; col_max: number; row_min: number; row_max: number };
  pitch: { x_px: number; y_px: number };
  origin: { x_px: number; y_px: number };
  visible_cols: number;
  visible_rows: number;
};

type Props = {
  onExit: () => void;
};

export function CompareDetectors({ onExit }: Props) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [idx, setIdx] = useState(0);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [cnn, setCnn] = useState<Circle[]>([]);
  const [sgf, setSgf] = useState<SgfResult | null>(null);
  const [edges, setEdges] = useState<EdgeProbs | null>(null);
  const [gridResult, setGridResult] = useState<GridResult | null>(null);
  const [pipeline, setPipeline] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [ingestProgress, setIngestProgress] = useState<{
    page: number; total: number; tasks: number;
  } | null>(null);
  const [peakThresh, setPeakThresh] = useState(0.3);
  const peakRef = useRef(peakThresh);
  peakRef.current = peakThresh;

  const current: Task | null = tasks[idx] ?? null;

  const loadTasks = async () => {
    const r = await fetch('/api/training/stone-tasks');
    if (!r.ok) return;
    const data = await r.json();
    const auto = (data.tasks as Task[]).filter((t) => t.source === 'auto_detected');
    setTasks(auto);
    setIdx(0);
  };

  useEffect(() => { loadTasks(); }, []);

  const fetchAll = async (taskId: string) => {
    setLoading(true);
    try {
      const thresh = peakRef.current;
      const [cRes, sRes, eRes, gRes, pRes] = await Promise.all([
        fetch(`/api/training/task-stones-cnn/${taskId}?peak_thresh=${thresh}&_t=${Date.now()}`, {
          cache: 'no-store',
        }),
        fetch(`/api/training/task-sgf/${taskId}?peak_thresh=${thresh}&_t=${Date.now()}`, {
          cache: 'no-store',
        }),
        fetch(`/api/training/task-edges/${taskId}?_t=${Date.now()}`, {
          cache: 'no-store',
        }),
        fetch(`/api/training/task-grid/${taskId}?_t=${Date.now()}`, {
          cache: 'no-store',
        }),
        fetch(`/api/training/task-pipeline/${taskId}?peak_thresh=${thresh}&_t=${Date.now()}`, {
          cache: 'no-store',
        }),
      ]);
      if (cRes.ok) {
        const d = await cRes.json();
        setCnn(
          (d.stones as Array<{ x: number; y: number; r: number; color: 'B' | 'W'; conf: number }>).map(
            (s) => ({ x: s.x, y: s.y, r: s.r, color: s.color, conf: s.conf }),
          ),
        );
      } else if (cRes.status === 503) {
        setCnn([]);
        setStatus('CNN model not trained yet.');
      } else {
        setCnn([]);
      }
      if (sRes.ok) setSgf(await sRes.json());
      else setSgf(null);
      if (eRes.ok) setEdges(await eRes.json());
      else setEdges(null);
      if (gRes.ok) setGridResult(await gRes.json());
      else setGridResult(null);
      if (pRes.ok) setPipeline(await pRes.json());
      else setPipeline(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!current) {
      setImgSize(null);
      setCnn([]);
      setSgf(null);
      setEdges(null);
      return;
    }
    setImgSize(null);
    setCnn([]);
    setSgf(null);
    setEdges(null);
    setGridResult(null);
    setPipeline(null);
    const img = new Image();
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = `/api/training/task-crops/${current.task_id}.png`;
    fetchAll(current.task_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.task_id]);

  const peakKey = useMemo(() => String(peakThresh), [peakThresh]);
  useEffect(() => {
    if (!current) return;
    const t = setTimeout(() => { fetchAll(current.task_id); }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [peakKey]);

  const skip = () => { setIdx((i) => i + 1); setStatus(null); };
  const prev = () => setIdx((i) => Math.max(0, i - 1));

  const clearPdf = async () => {
    if (!confirm('Clear the currently loaded PDF data?')) return;
    setClearing(true);
    try {
      const r = await fetch('/api/training/clear-stone-tasks', { method: 'POST' });
      if (!r.ok) throw new Error(`clear failed: ${r.status}`);
      const d = await r.json();
      setStatus(`Cleared ${d.removed} boards.`);
      setTasks([]);
      setIdx(0);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setClearing(false);
    }
  };

  const ingestPdf = async (file: File) => {
    setIngesting(true);
    setIngestProgress({ page: 0, total: 0, tasks: 0 });
    setStatus(`Ingesting ${file.name}…`);
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      const r = await fetch('/api/training/ingest-pdf-for-stones', {
        method: 'POST', body: form,
      });
      if (!r.ok) throw new Error(`ingest failed: ${r.status}`);
      if (!r.body) throw new Error('no response body');
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let totalPages = 0;
      let tasksAdded = 0;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.error) throw new Error(ev.error);
          if (ev.total_pages !== undefined) totalPages = ev.total_pages;
          if (ev.page !== undefined) {
            tasksAdded = ev.tasks_added ?? tasksAdded;
            setIngestProgress({ page: ev.page, total: totalPages, tasks: tasksAdded });
          }
          if (ev.done) tasksAdded = ev.total_tasks ?? tasksAdded;
        }
      }
      setStatus(`Ingested ${tasksAdded} boards from ${file.name}.`);
      await loadTasks();
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setIngesting(false);
      setIngestProgress(null);
    }
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') prev();
      if (e.key === 'ArrowRight') skip();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const renderStoneOverlay = (circles: Circle[]) => {
    if (!imgSize) return null;
    return (
      <svg
        viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
        className="stone-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {circles.map((c, i) => {
          const cls =
            c.color === 'B' ? 'circle-black'
            : c.color === 'W' ? 'circle-white'
            : 'circle-unassigned';
          return (
            <circle key={i} cx={c.x} cy={c.y} r={c.r} className={cls} />
          );
        })}
      </svg>
    );
  };

  // Color the 4 crop borders by edge-classifier confidence.
  // Green = high confidence it's a real board edge; red = high confidence
  // it isn't; yellow = in-between.
  const edgeBorderStyle = (p: number | undefined) => {
    if (p === undefined) return 'transparent';
    if (p >= 0.7) return 'rgba(0,180,0,0.9)';
    if (p <= 0.3) return 'rgba(220,40,40,0.9)';
    return 'rgba(235,180,0,0.9)';
  };

  return (
    <div className="stone-labeler">
      <div className="stone-toolbar">
        <label className="ingest-btn">
          {ingesting ? 'Ingesting…' : 'Ingest PDF'}
          <input
            type="file"
            accept="application/pdf"
            disabled={ingesting}
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) ingestPdf(f);
              e.target.value = '';
            }}
          />
        </label>
        <button
          onClick={clearPdf}
          disabled={clearing || ingesting || tasks.length === 0}
        >
          {clearing ? 'Clearing…' : 'Clear PDF'}
        </button>
        <div className="stone-progress">
          {tasks.length > 0
            ? `Board ${idx + 1} of ${tasks.length}${loading ? ' (detecting…)' : ''}`
            : 'No PDF ingested.'}
          {' · '}stones: {cnn.length}
        </div>
        <div className="stone-actions">
          <button onClick={prev} disabled={tasks.length === 0 || idx === 0}>◀</button>
          <button onClick={skip} disabled={tasks.length === 0}>Skip ▶</button>
          <button onClick={onExit}>Done</button>
        </div>
      </div>

      {ingestProgress && (
        <div className="ingest-progress">
          <div className="ingest-progress-label">
            Page {ingestProgress.page} of {ingestProgress.total || '?'} ·
            {' '}{ingestProgress.tasks} boards detected
          </div>
          <div className="ingest-progress-bar">
            <div
              className="ingest-progress-fill"
              style={{
                width: ingestProgress.total
                  ? `${(ingestProgress.page / ingestProgress.total) * 100}%`
                  : '0%',
              }}
            />
          </div>
        </div>
      )}

      <details className="detection-settings">
        <summary>CNN settings</summary>
        <div className="settings-grid">
          <label>
            <span>Peak threshold <em>{peakThresh.toFixed(2)}</em> (↓ = more recall)</span>
            <input
              type="range"
              step="0.02"
              min="0.05"
              max="0.9"
              value={peakThresh}
              onChange={(e) => setPeakThresh(Number(e.target.value))}
            />
          </label>
        </div>
      </details>

      {status && !ingestProgress && <div className="stone-status">{status}</div>}

      <div className="stone-stage">
        {current && (
          <div className="compare-body">
            <div className="compare-pair">
              <div className="compare-col">
                <div className="compare-label">Original (bbox + edges)</div>
                <div
                  className="stone-panel edge-boxed"
                  style={{
                    borderTopColor: edgeBorderStyle(edges?.top),
                    borderBottomColor: edgeBorderStyle(edges?.bottom),
                    borderLeftColor: edgeBorderStyle(edges?.left),
                    borderRightColor: edgeBorderStyle(edges?.right),
                  }}
                >
                  <img
                    src={`/api/training/task-crops/${current.task_id}.png`}
                    alt="board crop"
                    className="stone-img"
                  />
                </div>
                {edges && (
                  <div className="edge-probs">
                    L {edges.left.toFixed(2)} · R {edges.right.toFixed(2)} ·
                    {' '}T {edges.top.toFixed(2)} · B {edges.bottom.toFixed(2)}
                  </div>
                )}
              </div>
              <div className="compare-col">
                <div className="compare-label">CNN stones</div>
                <div className="stone-panel">
                  <img
                    src={`/api/training/task-crops/${current.task_id}.png`}
                    alt="cnn overlay"
                    className="stone-img"
                  />
                  {renderStoneOverlay(cnn)}
                </div>
              </div>
            </div>
            {pipeline && (
              <div className="compare-reconstructed">
                <div className="compare-label">
                  Reconstructed SGF (4-model pipeline) — edges:
                  {' '}{Object.entries(pipeline.edges).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none'}
                  {' · '}window cols {pipeline.window.col_min}-{pipeline.window.col_max}
                  {', '}rows {pipeline.window.row_min}-{pipeline.window.row_max}
                  {' · '}pitch ({pipeline.pitch.x_px.toFixed(1)},{pipeline.pitch.y_px.toFixed(1)})
                </div>
                <div className="reconstructed-board-large">
                  <Board
                    stones={pipeline.stones.map(
                      (s) => ({ x: s.col, y: s.row, color: s.color }) as Stone,
                    )}
                    onPlay={() => {}}
                  />
                </div>
              </div>
            )}
          </div>
        )}
        {!current && tasks.length === 0 && (
          <div className="stone-empty">
            Ingest a PDF to see bbox + edge detection + CNN stone detection + SGF reconstruction.
          </div>
        )}
      </div>
    </div>
  );
}
