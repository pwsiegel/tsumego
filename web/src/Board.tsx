import type { Stone } from './types';
import { BOARD_SIZE } from './types';
import './Board.css';

type Props = {
  stones: Stone[];
  onPlay: (x: number, y: number) => void;
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

export function Board({ stones, onPlay }: Props) {
  const occupied = new Map<string, Stone>();
  for (const s of stones) occupied.set(`${s.x},${s.y}`, s);

  return (
    <svg
      className="board"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
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
            className={filled ? 'hit occupied' : 'hit'}
            onClick={() => !filled && onPlay(x, y)}
          />
        );
      })}

      {/* Stones */}
      {stones.map((s) => (
        <g key={`stone-${s.x}-${s.y}`}>
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
