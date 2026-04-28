/**
 * A solve attempt is an ordered list of move points; the same intersection
 * may appear more than once when stones get captured and replayed in a
 * complex variation. The board only paints the *first* number played at
 * each intersection; any later moves at the same spot are reported as a
 * chain like "1...5...8" off to the side.
 */

export type MovePoint = { x: number; y: number };

export type NumberedMove = { x: number; y: number; number: number };

export type NumberedOverlay = {
  /** One entry per unique intersection: the first move number played there. */
  boardNumbers: NumberedMove[];
  /** Each chain has length ≥ 2 — the first number is the one painted on the
   * board, the rest are recaptures that landed on the same point. */
  chains: number[][];
};

export function computeNumberedOverlay(moves: MovePoint[]): NumberedOverlay {
  const chainAtPoint = new Map<string, number[]>();
  const firstAtPoint = new Map<string, number>();
  moves.forEach((m, i) => {
    const num = i + 1;
    const key = `${m.x},${m.y}`;
    const existing = chainAtPoint.get(key);
    if (existing === undefined) {
      chainAtPoint.set(key, [num]);
      firstAtPoint.set(key, num);
    } else {
      existing.push(num);
    }
  });
  const boardNumbers: NumberedMove[] = [];
  for (const [key, num] of firstAtPoint) {
    const [x, y] = key.split(',').map(Number);
    boardNumbers.push({ x, y, number: num });
  }
  const chains = Array.from(chainAtPoint.values()).filter((c) => c.length > 1);
  return { boardNumbers, chains };
}
