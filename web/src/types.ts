export type Color = 'B' | 'W';

export type Stone = {
  x: number;
  y: number;
  color: Color;
  // Present for numbered moves, absent for initial-position setup stones.
  number?: number;
};

export const BOARD_SIZE = 19;

export const other = (c: Color): Color => (c === 'B' ? 'W' : 'B');
