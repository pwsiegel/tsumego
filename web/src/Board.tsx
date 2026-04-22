import type { Stone } from './types';
import { BOARD_SIZE } from './types';
import './Board.css';

type Props = {
  stones: Stone[];
  /** Called with the intersection clicked and whether Shift was held,
   * so callers can map unmodified vs shifted clicks to different
   * behaviors (e.g. place B vs W). */
  onPlay: (x: number, y: number, shift: boolean) => void;
  /** When true, clicks fire on every intersection (including occupied
   * ones) so the caller can cycle / remove / overwrite. Default false
   * keeps "place a stone" semantics. */
  editable?: boolean;
  /** Optional viewport to show only a rectangular region of the 19x19.
   * Useful when the caller wants to zoom to the stones of interest. */
  viewport?: { colMin: number; colMax: number; rowMin: number; rowMax: number };
  /** Display-only mode: suppress the hover darkening on empty cells so
   * the board truthfully represents its stones without mouse artifacts.
   * Click handlers are still wired up but no visual hover feedback. */
  displayOnly?: boolean;
  /** Render 0-indexed column numbers above and row numbers to the left
   * of the grid, so a viewer can refer to positions unambiguously when
   * describing what they see. */
  showCoords?: boolean;
};

const PADDING = 30;
const CELL = 32;
const SIZE = PADDING * 2 + CELL * (BOARD_SIZE - 1);
const STONE_R = CELL * 0.47;

const STAR_POINTS: Array<[number, number]> = [
  [3, 3], [3, 9], [3, 15],
  [9, 3], [9, 9], [9, 15],
  [15, 3], [15, 9], [15, 15],
];

const toPx = (i: number) => PADDING + i * CELL;

export function Board({
  stones, onPlay, editable = false, viewport,
  displayOnly = false, showCoords = false,
}: Props) {
  const occupied = new Map<string, Stone>();
  for (const s of stones) occupied.set(`${s.x},${s.y}`, s);

  // If a viewport is given, tighten the viewBox around those columns/rows.
  // Extend to the board's outer padding if a side of the viewport is at
  // the board edge (so the outer grid line + padding are visible), else
  // leave a modest buffer inside.
  let vb = `0 0 ${SIZE} ${SIZE}`;
  if (viewport) {
    const BUF = CELL * 0.7;
    const vx0 = viewport.colMin <= 0 ? 0 : toPx(viewport.colMin) - BUF;
    const vy0 = viewport.rowMin <= 0 ? 0 : toPx(viewport.rowMin) - BUF;
    const vx1 = viewport.colMax >= BOARD_SIZE - 1 ? SIZE : toPx(viewport.colMax) + BUF;
    const vy1 = viewport.rowMax >= BOARD_SIZE - 1 ? SIZE : toPx(viewport.rowMax) + BUF;
    vb = `${vx0} ${vy0} ${vx1 - vx0} ${vy1 - vy0}`;
  }

  return (
    <svg
      className={`board${displayOnly ? ' display-only' : ''}`}
      viewBox={vb}
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect x={0} y={0} width={SIZE} height={SIZE} className="board-bg" />

      {/* Grid lines */}
      {Array.from({ length: BOARD_SIZE }, (_, i) => (
        <g key={`grid-${i}`}>
          <line
            x1={toPx(i)} y1={toPx(0)}
            x2={toPx(i)} y2={toPx(BOARD_SIZE - 1)}
            className="grid-line"
          />
          <line
            x1={toPx(0)} y1={toPx(i)}
            x2={toPx(BOARD_SIZE - 1)} y2={toPx(i)}
            className="grid-line"
          />
        </g>
      ))}

      {/* Coordinate labels (0-indexed to match internal col/row) */}
      {showCoords && Array.from({ length: BOARD_SIZE }, (_, i) => (
        <g key={`coord-${i}`} className="coord-label">
          <text
            x={toPx(i)} y={PADDING - 12}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={11}
          >{i}</text>
          <text
            x={PADDING - 12} y={toPx(i)}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={11}
          >{i}</text>
        </g>
      ))}

      {/* Star points */}
      {STAR_POINTS.map(([x, y]) => (
        <circle
          key={`star-${x}-${y}`}
          cx={toPx(x)} cy={toPx(y)}
          r={3.5}
          className="star-point"
        />
      ))}

      {/* Click targets */}
      {Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, i) => {
        const x = i % BOARD_SIZE;
        const y = Math.floor(i / BOARD_SIZE);
        const key = `${x},${y}`;
        const filled = occupied.has(key);
        return (
          <rect
            key={`hit-${key}`}
            x={toPx(x) - CELL / 2}
            y={toPx(y) - CELL / 2}
            width={CELL}
            height={CELL}
            className={filled && !editable ? 'hit occupied' : 'hit'}
            onClick={(e) => (editable || !filled) && onPlay(x, y, e.shiftKey)}
          />
        );
      })}

      {/* Stones */}
      {stones.map((s) => (
        <g key={`stone-${s.x}-${s.y}`} style={{ pointerEvents: 'none' }}>
          <circle
            cx={toPx(s.x)} cy={toPx(s.y)}
            r={STONE_R}
            className={s.color === 'B' ? 'stone-black' : 'stone-white'}
          />
          {s.number !== undefined && (
            <text
              x={toPx(s.x)} y={toPx(s.y)}
              className={s.color === 'B' ? 'label-on-black' : 'label-on-white'}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={CELL * 0.5}
            >
              {s.number}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
