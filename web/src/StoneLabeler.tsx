import { useEffect, useMemo, useRef, useState } from 'react';
import './StoneLabeler.css';

type Task = {
  task_id: string;
  source: string;
  labeled: boolean;
};

type Color = 'B' | 'W';

type LabeledCircle = {
  x: number;
  y: number;
  r: number;
  color: Color | null; // null = unassigned / detection suggestion
};

type Settings = {
  min_r_frac: number;
  max_r_frac: number;
  hough_param2: number;
  white_ring_thresh: number;
};

type Props = {
  onExit: () => void;
};

export function StoneLabeler({ onExit }: Props) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [idx, setIdx] = useState(0);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [circles, setCircles] = useState<LabeledCircle[]>([]);
  const [phase, setPhase] = useState<Color>('B');
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState<{
    page: number; total: number; tasks: number;
  } | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    min_r_frac: 0.02,
    max_r_frac: 0.15,
    hough_param2: 40,
    white_ring_thresh: 0.1,
  });
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const svgRef = useRef<SVGSVGElement>(null);

  const current: Task | null = tasks[idx] ?? null;

  const medianR = useMemo(() => {
    const rs = circles.map((c) => c.r).sort((a, b) => a - b);
    if (rs.length) return rs[Math.floor(rs.length / 2)];
    return imgSize ? Math.min(imgSize.w, imgSize.h) * 0.04 : 20;
  }, [circles, imgSize]);

  const loadTasks = async () => {
    const r = await fetch('/api/training/stone-tasks');
    if (!r.ok) return;
    const data = await r.json();
    const auto = (data.tasks as Task[]).filter((t) => t.source === 'auto_detected');
    for (let i = auto.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [auto[i], auto[j]] = [auto[j], auto[i]];
    }
    setTasks(auto);
    setIdx(0);
  };

  useEffect(() => { loadTasks(); }, []);

  const fetchCircles = async (taskId: string) => {
    setDetecting(true);
    const s = settingsRef.current;
    const q = new URLSearchParams({
      min_r_frac: String(s.min_r_frac),
      max_r_frac: String(s.max_r_frac),
      hough_param2: String(s.hough_param2),
      white_ring_thresh: String(s.white_ring_thresh),
      _t: String(Date.now()),
    });
    try {
      const r = await fetch(`/api/training/task-circles/${taskId}?${q}`, {
        cache: 'no-store',
      });
      if (!r.ok) return;
      const data = await r.json();
      setCircles(
        (data.circles as Array<{ x: number; y: number; r: number; color?: 'B' | 'W' | null }>).map((c) => ({
          x: c.x, y: c.y, r: c.r,
          color: (c.color === 'B' || c.color === 'W') ? c.color : null,
        }))
      );
    } finally {
      setDetecting(false);
    }
  };

  useEffect(() => {
    if (!current) {
      setImgSize(null);
      setCircles([]);
      return;
    }
    setImgSize(null);
    setCircles([]);
    setPhase('B');
    const img = new Image();
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = `/api/training/task-crops/${current.task_id}.png`;
    fetchCircles(current.task_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.task_id]);

  // Debounced re-detect when settings change (stays on the current board).
  const settingsKey = useMemo(
    () => `${settings.min_r_frac}-${settings.max_r_frac}-${settings.hough_param2}-${settings.white_ring_thresh}`,
    [settings],
  );
  useEffect(() => {
    if (!current) return;
    const t = setTimeout(() => {
      fetchCircles(current.task_id);
    }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settingsKey]);

  const svgPoint = (ev: { clientX: number; clientY: number }) => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const ctm = svg.getScreenCTM()!.inverse();
    const p = pt.matrixTransform(ctm);
    return { x: p.x, y: p.y };
  };

  // Click behavior:
  //   hit a circle of the current phase's color → delete it entirely
  //   hit any other circle (different color / unassigned) → reclassify as
  //     current phase
  //   hit blank area → add a new circle at click, assigned to current phase
  const handleClick = (ev: React.MouseEvent) => {
    if (!imgSize) return;
    const p = svgPoint(ev);
    const hitIdx = circles.findIndex(
      (c) => Math.hypot(c.x - p.x, c.y - p.y) < c.r,
    );
    if (hitIdx >= 0) {
      const c = circles[hitIdx];
      if (c.color === phase) {
        setCircles((prev) => prev.filter((_, i) => i !== hitIdx));
      } else {
        setCircles((prev) => prev.map((c, i) =>
          i === hitIdx ? { ...c, color: phase } : c,
        ));
      }
    } else {
      setCircles((prev) => [
        ...prev,
        { x: p.x, y: p.y, r: medianR, color: phase },
      ]);
    }
  };

  // Right-click / shift-click deletes a circle outright (not just unassigns).
  const handleContextMenu = (ev: React.MouseEvent) => {
    ev.preventDefault();
    if (!imgSize) return;
    const p = svgPoint(ev);
    const hitIdx = circles.findIndex(
      (c) => Math.hypot(c.x - p.x, c.y - p.y) < c.r,
    );
    if (hitIdx >= 0) {
      setCircles((prev) => prev.filter((_, i) => i !== hitIdx));
    }
  };

  const saveCurrent = async () => {
    if (!current) return;
    const black = circles.filter((c) => c.color === 'B');
    const white = circles.filter((c) => c.color === 'W');
    if (black.length === 0 && white.length === 0) {
      if (!confirm('Nothing labeled — save as empty board?')) return;
    }
    setSaving(true);
    setStatus(null);
    try {
      const form = new FormData();
      form.append('task_id', current.task_id);
      form.append('black', JSON.stringify(black.map((c) => [c.x, c.y])));
      form.append('white', JSON.stringify(white.map((c) => [c.x, c.y])));
      const r = await fetch('/api/training/save-stone-points', {
        method: 'POST', body: form,
      });
      if (!r.ok) throw new Error(`save failed: ${r.status}`);
      const data = await r.json();
      setStatus(
        `Saved ${data.black_count}B / ${data.white_count}W. ` +
        `Totals: ${data.totals.labeled_tasks} boards, ` +
        `${data.totals.black} B / ${data.totals.white} W.`
      );
      setTasks((prev) => prev.map((t) =>
        t.task_id === current.task_id ? { ...t, labeled: true } : t,
      ));
      setIdx((i) => i + 1);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  const skip = () => { setIdx((i) => i + 1); setStatus(null); };
  const prev = () => setIdx((i) => Math.max(0, i - 1));

  const advance = () => {
    if (phase === 'B') setPhase('W');
    else saveCurrent();
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

  // Keyboard shortcuts.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.key === 's' || e.key === 'S') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        saveCurrent();
        return;
      }
      if (e.key === 'Enter') { e.preventDefault(); advance(); return; }
      if (e.key === 'ArrowLeft') prev();
      if (e.key === 'ArrowRight') skip();
      if (e.key === 'b' || e.key === 'B') setPhase('B');
      if (e.key === 'w' || e.key === 'W') setPhase('W');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, circles, phase]);

  const countB = circles.filter((c) => c.color === 'B').length;
  const countW = circles.filter((c) => c.color === 'W').length;
  const countUnassigned = circles.filter((c) => c.color === null).length;

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
        <div className={`phase-indicator phase-${phase.toLowerCase()}`}>
          Placing {phase === 'B' ? 'BLACK' : 'WHITE'} stones
        </div>
        <div className="stone-progress">
          {tasks.length > 0
            ? `Board ${idx + 1} of ${tasks.length}${detecting ? ' (detecting…)' : ''}`
            : 'No PDF ingested.'}
          {' · '}{countB}B / {countW}W{countUnassigned ? ` · ${countUnassigned} unclassified` : ''}
        </div>
        <div className="stone-actions">
          <button onClick={prev} disabled={tasks.length === 0 || idx === 0}>◀</button>
          <button onClick={skip} disabled={tasks.length === 0}>Skip ▶</button>
          <button
            onClick={advance}
            disabled={!current || saving}
            className="primary"
          >
            {saving ? 'Saving…' : phase === 'B' ? 'Next: white (Enter)' : 'Save & next (Enter)'}
          </button>
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
            <div className="ingest-progress-fill"
                 style={{ width: ingestProgress.total
                   ? `${(ingestProgress.page / ingestProgress.total) * 100}%` : '0%' }}/>
          </div>
        </div>
      )}

      <details className="detection-settings">
        <summary>Detection settings</summary>
        <div className="settings-grid">
          <label>
            <span>Min radius % <em>{(settings.min_r_frac * 100).toFixed(1)}%</em></span>
            <input type="range" step="0.005" min="0.01" max="0.3"
              value={settings.min_r_frac}
              onChange={(e) => setSettings((s) => ({ ...s, min_r_frac: Number(e.target.value) }))} />
          </label>
          <label>
            <span>Max radius % <em>{(settings.max_r_frac * 100).toFixed(1)}%</em></span>
            <input type="range" step="0.01" min="0.05" max="0.5"
              value={settings.max_r_frac}
              onChange={(e) => setSettings((s) => ({ ...s, max_r_frac: Number(e.target.value) }))} />
          </label>
          <label>
            <span>Hough sensitivity <em>{settings.hough_param2}</em> (↓ = more circles)</span>
            <input type="range" step="1" min="8" max="60"
              value={settings.hough_param2}
              onChange={(e) => setSettings((s) => ({ ...s, hough_param2: Number(e.target.value) }))} />
          </label>
          <label>
            <span>Outline coverage <em>{settings.white_ring_thresh.toFixed(2)}</em></span>
            <input type="range" step="0.01" min="0.1" max="1.0"
              value={settings.white_ring_thresh}
              onChange={(e) => setSettings((s) => ({ ...s, white_ring_thresh: Number(e.target.value) }))} />
          </label>
        </div>
      </details>

      {status && !ingestProgress && <div className="stone-status">{status}</div>}

      <div className="stone-stage">
        {current && (
          <div className="stone-pair">
            <div className="stone-panel">
              <img
                src={`/api/training/task-crops/${current.task_id}.png`}
                alt="board crop"
                className="stone-img"
              />
              {imgSize && (
                <svg
                  ref={svgRef}
                  viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
                  className="stone-overlay"
                  preserveAspectRatio="none"
                  onClick={handleClick}
                  onContextMenu={handleContextMenu}
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
              )}
            </div>
            <div className="stone-panel">
              <img
                src={`/api/training/task-crops/${current.task_id}.png`}
                alt="reference (unannotated)"
                className="stone-img"
              />
            </div>
          </div>
        )}
        {!current && tasks.length === 0 && (
          <div className="stone-empty">
            Ingest a PDF to start. Each detected board will appear with
            auto-detected circles in orange — click to label them (black
            first, then Enter for white). Click blank areas to add missed
            stones. Right-click / Shift-click removes a circle entirely.
          </div>
        )}
      </div>
    </div>
  );
}
