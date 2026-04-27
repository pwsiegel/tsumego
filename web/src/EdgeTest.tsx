import { useEffect, useRef, useState } from 'react';
import { api, type BoardListItem, type BoardTJunctionEdges, type Junction } from './api';
import './EdgeTest.css';

type Props = {
  onExit: () => void;
};

const SIDES = ['left', 'right', 'top', 'bottom'] as const;
type Side = typeof SIDES[number];

const KIND_COLOR: Record<Junction['kind'], string> = {
  'T': 'rgb(0, 150, 60)',
  'L': 'rgb(20, 110, 200)',
  '+': 'rgb(200, 80, 60)',
  'I': 'rgb(160, 160, 160)',
  '?': 'rgb(120, 120, 120)',
};

type Aggregate = {
  total_boards: number;
  by_side: Record<Side, number>;
  by_count: Record<0 | 1 | 2 | 3 | 4, number>;
};

export function EdgeTest({ onExit }: Props) {
  const [boards, setBoards] = useState<BoardListItem[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [result, setResult] = useState<BoardTJunctionEdges | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [peakThresh, setPeakThresh] = useState<number>(0.3);
  const [status, setStatus] = useState<string | null>(null);
  const [aggregate, setAggregate] = useState<Aggregate | null>(null);
  const [aggregating, setAggregating] = useState(false);
  const inFlight = useRef<string | null>(null);
  const aggCancel = useRef<boolean>(false);

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
        const r = await api.pdf.detectTJunctionEdges(item.page_idx, item.bbox_idx, thresh);
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
      setAggregate(null);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      setUploading(false);
    }
  };

  const runAggregate = async () => {
    if (boards.length === 0 || aggregating) return;
    aggCancel.current = false;
    setAggregating(true);
    const agg: Aggregate = {
      total_boards: 0,
      by_side: { left: 0, right: 0, top: 0, bottom: 0 },
      by_count: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 },
    };
    for (let i = 0; i < boards.length; i++) {
      if (aggCancel.current) break;
      setStatus(`Aggregating ${i + 1}/${boards.length}…`);
      try {
        const r = await api.pdf.detectTJunctionEdges(
          boards[i].page_idx, boards[i].bbox_idx, peakThresh,
        );
        agg.total_boards++;
        let n = 0;
        for (const s of SIDES) {
          if (r.edges[s]) {
            agg.by_side[s]++;
            n++;
          }
        }
        agg.by_count[n as 0 | 1 | 2 | 3 | 4]++;
      } catch {
        // skip failures
      }
    }
    setAggregate(agg);
    setAggregating(false);
    setStatus(aggCancel.current
      ? `Aggregation cancelled at ${agg.total_boards}/${boards.length}.`
      : `Aggregation done over ${agg.total_boards} boards.`);
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

    // Show only T's and L's (the kinds that vote on edges). Each gets a
    // dot plus a small arrow pointing in each outward direction.
    const arrowLen = Math.min(W, H) * 0.022;
    const dirVec: Record<string, [number, number]> = {
      N: [0, -1], E: [1, 0], S: [0, 1], W: [-1, 0],
    };
    const arrow = (j: Junction, dir: string, idx: number) => {
      const [dx, dy] = dirVec[dir];
      return (
        <line
          key={idx}
          x1={j.x} y1={j.y}
          x2={j.x + dx * arrowLen} y2={j.y + dy * arrowLen}
          stroke={KIND_COLOR[j.kind]} strokeWidth={r * 0.6} opacity={0.95}
        />
      );
    };
    const voting = result.junctions.filter(
      (j) => j.kind === 'T' || j.kind === 'L',
    );

    const stoneTickColor = 'rgb(255, 140, 0)';
    const stoneRingColor = 'rgba(255, 140, 0, 0.55)';
    const stoneDirVec: Record<'N' | 'E' | 'S' | 'W', [number, number]> = {
      N: [0, -1], E: [1, 0], S: [0, 1], W: [-1, 0],
    };

    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="edge-overlay"
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      >
        {voting.map((j, i) => (
          <g key={`j${i}`}>
            {j.outward.map((d, k) => arrow(j, d, k))}
            <circle cx={j.x} cy={j.y} r={r * 0.9}
                    fill={KIND_COLOR[j.kind]} opacity={0.95} />
          </g>
        ))}

        {/* Per-stone edge classification: a stone gets an outward tick
            on each side that locally tested as "no neighbor stone +
            no grid ink" → on the board edge. Ringed if any side
            fires, so isolated stones with all-edge-everywhere can be
            distinguished from interior. */}
        {result.stone_edges.map((s, i) => {
          const fires = (['N', 'E', 'S', 'W'] as const).filter((d) => s.sides[d]);
          if (fires.length === 0) return null;
          const tickInner = s.r * 1.1;
          const tickOuter = s.r * 2.2;
          return (
            <g key={`se${i}`}>
              <circle cx={s.x} cy={s.y} r={s.r * 1.15}
                      fill="none" stroke={stoneRingColor}
                      strokeWidth={r * 0.4} />
              {fires.map((d, k) => {
                const [dx, dy] = stoneDirVec[d];
                return (
                  <line
                    key={k}
                    x1={s.x + dx * tickInner} y1={s.y + dy * tickInner}
                    x2={s.x + dx * tickOuter} y2={s.y + dy * tickOuter}
                    stroke={stoneTickColor} strokeWidth={r * 0.55}
                    strokeLinecap="round"
                  />
                );
              })}
            </g>
          );
        })}

        {/* Edge lines drawn through the actual T/L junctions that voted
            for each side. Median perpendicular position of the relevant
            junctions, with extent spanning their min..max parallel
            position. Nothing is drawn for sides that voted false. */}
        {(() => {
          const median = (xs: number[]) =>
            xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
          const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];

          for (const [side, dir] of [
            ['left', 'W'], ['right', 'E'], ['top', 'N'], ['bottom', 'S'],
          ] as const) {
            if (!result.edges[side]) continue;
            const js = result.junctions.filter(
              (j) => (j.kind === 'T' || j.kind === 'L') && j.outward.includes(dir),
            );
            if (js.length === 0) continue;
            if (side === 'left' || side === 'right') {
              const x = median(js.map((j) => j.x));
              const ys = js.map((j) => j.y);
              lines.push({ x1: x, y1: Math.min(...ys), x2: x, y2: Math.max(...ys) });
            } else {
              const y = median(js.map((j) => j.y));
              const xs = js.map((j) => j.x);
              lines.push({ x1: Math.min(...xs), y1: y, x2: Math.max(...xs), y2: y });
            }
          }
          return (
            <g>
              {lines.map((l, i) => (
                <line key={`edge${i}`} {...l}
                      stroke="rgb(0, 170, 200)" strokeWidth={r * 0.7} opacity={0.85} />
              ))}
            </g>
          );
        })()}
      </svg>
    );
  };

  const current = boards[safeSelected];
  const summary = result
    ? `edges ${SIDES.filter((s) => result.edges[s]).join('/') || '∅'}  ·  ${result.junctions.length} junctions`
    : '∅';

  return (
    <div className="edge-test">
      <div className="edge-toolbar">
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
        <button
          onClick={() => (aggregating ? (aggCancel.current = true) : runAggregate())}
          disabled={boards.length === 0}
        >
          {aggregating ? 'Cancel aggregate' : 'Run on all boards'}
        </button>
        <div className="edge-status">
          {boards.length === 0
            ? 'No PDF uploaded.'
            : current
              ? `Board ${safeSelected + 1} of ${boards.length}${loading ? ' (detecting…)' : ''}  ·  page ${current.page_idx + 1}, bbox ${current.bbox_idx}  ·  ${summary}`
              : ''}
        </div>
        <div className="edge-actions">
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

      {status && <div className="edge-message">{status}</div>}

      {current && (
        <div className="edge-stage">
          <div className="edge-panel">
            <img
              src={api.pdf.boardCleanedUrl(current.page_idx, current.bbox_idx)}
              alt={`page ${current.page_idx + 1} bbox ${current.bbox_idx}`}
              className="edge-img"
            />
            {renderOverlay()}
          </div>
          <div className="edge-summary">
            <h3>This board</h3>
            <table>
              <thead>
                <tr>
                  <th>side</th><th>edge?</th>
                  <th className="real">T</th><th className="real">L</th>
                </tr>
              </thead>
              <tbody>
                {result && SIDES.map((s) => {
                  const t = result.sides[s];
                  return (
                    <tr key={s}>
                      <td>{s}</td>
                      <td>{result.edges[s] ? '✓' : '·'}</td>
                      <td className="real">{t.t}</td>
                      <td className="real">{t.l}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {result && (() => {
              const sideDir: Record<Side, 'N' | 'E' | 'S' | 'W'> = {
                left: 'W', right: 'E', top: 'N', bottom: 'S',
              };
              return (
                <div style={{ marginTop: '0.6rem', fontSize: '0.85em' }}>
                  {SIDES.map((s) => {
                    const d = sideDir[s];
                    const voting = result.junctions.filter(
                      (j) => (j.kind === 'T' || j.kind === 'L') && j.outward.includes(d),
                    );
                    if (voting.length === 0) return null;
                    return (
                      <div key={s} style={{ marginTop: '0.2rem' }}>
                        <strong>{s}</strong>:&nbsp;
                        {voting.map((j, i) => (
                          <span key={i} style={{ marginRight: '0.5em' }}>
                            {j.kind}({j.x.toFixed(0)},{j.y.toFixed(0)})
                          </span>
                        ))}
                      </div>
                    );
                  })}
                </div>
              );
            })()}
            {aggregate && (
              <div className="agg">
                <h3 style={{ marginTop: '1rem' }}>All {aggregate.total_boards} boards</h3>
                <div>
                  edge hits — left {aggregate.by_side.left}, right {aggregate.by_side.right},
                  top {aggregate.by_side.top}, bottom {aggregate.by_side.bottom}
                </div>
                <div style={{ marginTop: '0.4rem' }}>
                  boards w/ # edges:&nbsp;
                  0={aggregate.by_count[0]}, 1={aggregate.by_count[1]},
                  2={aggregate.by_count[2]}, 3={aggregate.by_count[3]},
                  4={aggregate.by_count[4]}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
