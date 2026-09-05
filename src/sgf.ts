// Minimal SGF for full games: generate from a move list, and parse the main
// line back to coloured moves. App coords are (x from left, y from top), which
// maps directly onto SGF points (aa = top-left), so (15,3) -> "pd" (Q16).

import type { GameMove } from './data/model';

const A = 'a'.charCodeAt(0);
const point = (x: number, y: number) => String.fromCharCode(A + x) + String.fromCharCode(A + y);
const esc = (s: string) => s.replace(/([\]\\])/g, '\\$1');

export type SgfMeta = {
  boardSize?: number;
  komi?: number | null;   // null = omit the KM tag (unknown komi)
  rules?: string;         // '' = omit the RU tag (unknown ruleset)
  handicap?: number;      // HA (emitted when >= 2)
  setup?: GameMove[];     // AB/AW placement stones (handicap or free setup)
  name?: string;        // GN
  playerBlack?: string;
  playerWhite?: string;
  rankBlack?: string;
  rankWhite?: string;
  date?: string;        // DT, normally YYYY-MM-DD
  event?: string;       // EV, e.g. a tournament or the server it was played on
  result?: string;      // RE, e.g. "B+R"
};

export function toSgf(moves: GameMove[], meta: SgfMeta = {}): string {
  const {
    boardSize = 19, komi = 7.5, rules = 'Chinese', name = '',
    handicap = 0, setup = [],
    playerBlack = '', playerWhite = '', rankBlack = '', rankWhite = '',
    date = '', event = '', result = '',
  } = meta;
  const ab = setup.filter((s) => s.color === 'B');
  const aw = setup.filter((s) => s.color === 'W');
  const root = [
    'GM[1]', 'FF[4]', `SZ[${boardSize}]`,
    komi != null && `KM[${komi}]`,
    rules && `RU[${esc(rules)}]`,
    handicap >= 2 && `HA[${handicap}]`,
    ab.length > 0 && `AB${ab.map((s) => `[${point(s.x, s.y)}]`).join('')}`,
    aw.length > 0 && `AW${aw.map((s) => `[${point(s.x, s.y)}]`).join('')}`,
    name && `GN[${esc(name)}]`,
    event && `EV[${esc(event)}]`,
    playerBlack && `PB[${esc(playerBlack)}]`,
    playerWhite && `PW[${esc(playerWhite)}]`,
    rankBlack && `BR[${esc(rankBlack)}]`,
    rankWhite && `WR[${esc(rankWhite)}]`,
    date && `DT[${date}]`,
    result && `RE[${esc(result)}]`,
  ].filter(Boolean).join('');
  const body = moves.map((m) => `;${m.color}[${point(m.x, m.y)}]`).join('');
  return `(;${root}${body})`;
}

export type SgfInfo = {
  name: string;         // GN[] (empty when absent)
  playerBlack: string;
  playerWhite: string;
  rankBlack: string;
  rankWhite: string;
  date: string;         // DT[] (empty when absent)
  event: string;        // EV[] (empty when absent)
  result: string;       // RE[] value, e.g. "W+0.25" (empty when absent)
  boardSize: number | null;   // SZ[] (null when absent)
  komi: number | null;        // KM[] (null when absent)
  rules: string;              // RU[] (empty when absent)
  handicap: number | null;    // HA[] (null when absent)
  hasSetup: boolean;          // AB[]/AW[] present (handicap/problem setups)
};

/** Fox ranks use Chinese dan/kyu suffixes ("5段", "9级"); render them as
 * "5d" / "9k". No-op for ranks that don't use those characters. */
function normalizeRank(rank: string): string {
  return rank.replace(/段/g, 'd').replace(/[级級]/g, 'k');
}

/** Root metadata from an SGF (empty strings / nulls when absent). */
export function sgfInfo(sgf: string): SgfInfo {
  const prop = (key: string) => sgf.match(new RegExp(`\\b${key}\\[([^\\]]*)\\]`))?.[1] ?? '';
  const num = (s: string) => (s !== '' && Number.isFinite(Number(s)) ? Number(s) : null);
  return {
    name: prop('GN'),
    playerBlack: prop('PB'),
    playerWhite: prop('PW'),
    rankBlack: normalizeRank(prop('BR')),
    rankWhite: normalizeRank(prop('WR')),
    date: prop('DT'),
    event: prop('EV'),
    result: prop('RE'),
    boardSize: num(prop('SZ')),
    komi: num(prop('KM')),
    rules: prop('RU'),
    handicap: num(prop('HA')),
    hasSetup: /\bA[BW]\[/.test(sgf),
  };
}

/** Setup (placement) stones from an SGF's AB/AW properties — handicap stones
 * or a free setup. Scans the whole text (values may repeat: AB[dd][pp]). */
export function setupStonesFromSgf(sgf: string): GameMove[] {
  const out: GameMove[] = [];
  const re = /\bA([BW])((?:\s*\[[a-s][a-s]\])+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(sgf)) !== null) {
    const color = m[1] as GameMove['color'];
    for (const v of m[2].match(/\[[a-s][a-s]\]/g) ?? []) {
      out.push({ color, x: v.charCodeAt(1) - A, y: v.charCodeAt(2) - A });
    }
  }
  return out;
}

/** Standard 19x19 hoshi placement for an n-stone handicap (app coords). */
export function standardHandicapStones(n: number): GameMove[] {
  const pts: Record<number, [number, number][]> = {
    2: [[15, 3], [3, 15]],
    3: [[15, 3], [3, 15], [15, 15]],
    4: [[15, 3], [3, 15], [15, 15], [3, 3]],
    5: [[15, 3], [3, 15], [15, 15], [3, 3], [9, 9]],
    6: [[15, 3], [3, 15], [15, 15], [3, 3], [3, 9], [15, 9]],
    7: [[15, 3], [3, 15], [15, 15], [3, 3], [3, 9], [15, 9], [9, 9]],
    8: [[15, 3], [3, 15], [15, 15], [3, 3], [3, 9], [15, 9], [9, 3], [9, 15]],
    9: [[15, 3], [3, 15], [15, 15], [3, 3], [3, 9], [15, 9], [9, 3], [9, 15], [9, 9]],
  };
  return (pts[n] ?? []).map(([x, y]) => ({ color: 'B', x, y }));
}

/** Main-line coloured moves from an SGF (ignores setup stones, variations, and
 * passes). Tolerant of our own output; not a full SGF parser. */
export function movesFromSgf(sgf: string): GameMove[] {
  const out: GameMove[] = [];
  const re = /;\s*([BW])\[([a-s][a-s])\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(sgf)) !== null) {
    out.push({
      color: m[1] as GameMove['color'],
      x: m[2].charCodeAt(0) - A,
      y: m[2].charCodeAt(1) - A,
    });
  }
  return out;
}

/** Main-line moves of an arbitrary SGF: at every branch point only the first
 * variation is followed, and property values are skipped properly (comments
 * can contain parentheses). Use for uploaded SGFs, which may carry real
 * variation trees that the flat `movesFromSgf` scan would interleave. Passes
 * (empty or "tt" on <=19x19) are dropped, like `movesFromSgf`. */
export function mainlineMovesFromSgf(sgf: string): GameMove[] {
  const out: GameMove[] = [];
  const took: boolean[] = [];   // per open group: has its first child been taken?
  let skip = 0;                 // >0: inside a discarded sibling group
  let i = 0;
  while (i < sgf.length) {
    const ch = sgf[i];
    if (ch === '(') {
      if (skip > 0) { skip++; i++; continue; }
      const parent = took.length - 1;
      if (parent >= 0 && took[parent]) { skip = 1; i++; continue; }
      if (parent >= 0) took[parent] = true;
      took.push(false);
      i++;
    } else if (ch === ')') {
      if (skip > 0) skip--;
      else took.pop();
      i++;
    } else if (ch >= 'A' && ch <= 'Z') {
      let id = '';
      while (i < sgf.length && sgf[i] >= 'A' && sgf[i] <= 'Z') { id += sgf[i]; i++; }
      const vals: string[] = [];
      for (;;) {
        while (i < sgf.length && /\s/.test(sgf[i])) i++;
        if (sgf[i] !== '[') break;
        i++;
        let v = '';
        while (i < sgf.length && sgf[i] !== ']') {
          if (sgf[i] === '\\' && i + 1 < sgf.length) i++;
          v += sgf[i];
          i++;
        }
        i++;
        vals.push(v);
      }
      if (skip === 0 && (id === 'B' || id === 'W') && /^[a-s][a-s]$/.test(vals[0] ?? '') && vals[0] !== 'tt') {
        out.push({ color: id, x: vals[0].charCodeAt(0) - A, y: vals[0].charCodeAt(1) - A });
      }
    } else i++;
  }
  return out;
}

/** Set, replace, or (with '') remove root properties of an SGF in place —
 * the move record is left byte-identical, unlike regenerating via toSgf
 * (which would drop passes and variations). Values are escaped. */
export function patchSgfMeta(
  sgf: string,
  tags: Partial<Record<'GN' | 'PB' | 'PW' | 'BR' | 'WR' | 'DT' | 'EV' | 'RE' | 'KM' | 'RU', string>>,
): string {
  let out = sgf;
  for (const [key, raw] of Object.entries(tags)) {
    if (raw === undefined) continue;
    const value = raw.replace(/([\]\\])/g, '\\$1');
    const re = new RegExp(`\\b${key}\\[[^\\]]*\\]`);
    if (re.test(out)) out = out.replace(re, value === '' ? '' : `${key}[${value}]`);
    else if (value !== '') out = out.replace('(;', `(;${key}[${value}]`);
  }
  return out;
}
