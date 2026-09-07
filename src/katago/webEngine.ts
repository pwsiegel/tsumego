// App-facing analysis facade. Two runtimes behind one interface:
//   - browser models (tfjs/WebGPU, vendored engine) — nets in Firebase Storage
//   - a local model (the native KataGo backend via /api/katago) — dev only
// Both report winrate/score from Black's perspective, so they read directly
// through our B+/W+ scoreLabel.

import { getDownloadURL, ref } from 'firebase/storage';
import { getKataGoEngineClient, isKataGoCanceledError, getEnginePerf, getChosenBatchSize } from './engine/katago/client';
import { autoBatchSize } from './engine/katago/autoBatch';
import type { KataGoAnalysisPayload } from './engine/katago/types';
import { storage } from '../firebase';
import { analyze as backendAnalyze, type Analysis as BackendAnalysis, type AnalyzeParams } from '../data/katago';
import type { BoardState, Move } from './types';
import type { Color, Stone } from '../types';
import { BOARD_SIZE } from '../types';
import type { GameMove } from '../data/model';

export type AnalysisModel = {
  id: string;
  name: string;
  runtime: string;
  strength: string;        // short strength / speed descriptor for the UI
  kind: 'browser' | 'local';
  netPath?: string;        // browser only — Firebase Storage path
  defaultVisits: number;
};

export const BROWSER_MODELS: AnalysisModel[] = [
  { id: 'b18', name: 'kata1-b18c384nbt', runtime: 'WebGPU', strength: 'strong', kind: 'browser', netPath: 'katago/kata1-b18c384nbt.bin.gz', defaultVisits: 50 },
];

// The native KataGo backend (real engine) — only offered when it's reachable
// (dev with `make api`). Same net as the browser b18, full-strength search.
export const LOCAL_MODEL: AnalysisModel = {
  id: 'local', name: 'kata1-b18c384nbt', runtime: 'Metal (native)', strength: 'strong', kind: 'local', defaultVisits: 1000,
};

let webgpuAvailablePromise: Promise<boolean> | null = null;
/** Whether a real (non-software) WebGPU adapter is available — the signal for
 * whether analysis can run at all. Cached; safe to call repeatedly. Mirrors what
 * the worker's `tf.setBackend('webgpu')` will get, so a null/fallback adapter
 * here means the engine cannot start. */
export function webgpuAvailable(): Promise<boolean> {
  if (!webgpuAvailablePromise) {
    webgpuAvailablePromise = (async () => {
      try {
        const gpu = (navigator as unknown as {
          gpu?: { requestAdapter: () => Promise<{ isFallbackAdapter?: boolean } | null> };
        }).gpu;
        if (!gpu) return false;
        const adapter = await gpu.requestAdapter();
        return !!adapter && !adapter.isFallbackAdapter;
      } catch {
        return false;
      }
    })();
  }
  return webgpuAvailablePromise;
}

/** Auto GPU batch size for the running browser engine, from its last on-load
 * forward-pass measurement (a safe middle before any measurement exists). This
 * is the value the worker also picks for a search when batchSize is omitted. */
export function recommendedBatchSize(): number {
  return autoBatchSize(getEnginePerf());
}

/** The batch size the worker most recently searched with (auto or manual), or
 * null before any browser analysis has run — for surfacing the live value in UI. */
export function activeBatchSize(): number | null {
  return getChosenBatchSize();
}

export type WebCandidate = {
  x: number;
  y: number;
  winrate: number;      // 0..1, Black
  scoreLead: number;    // points, Black perspective
  visits: number;
  pointsLost: number;   // points behind the best move (0 = best), perspective-free
  order: number;
};

export type WebAnalysis = {
  rootWinrate: number;      // 0..1, Black
  rootScoreLead: number;    // points, Black perspective
  rootVisits: number;
  policyTop: { x: number; y: number } | null;   // net's raw top move (pre-search)
  moves: WebCandidate[];    // best-first
  // Eval of a specifically-requested move (the game's next move), even when the
  // search never visited it. scoreLead is Black-perspective.
  playedEval?: { scoreLead: number; pointsLost: number } | null;
};

const toPlayer = (c: Color): 'black' | 'white' => (c === 'B' ? 'black' : 'white');

function toBoardState(stones: Stone[]): BoardState {
  const board: BoardState = Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => null),
  );
  for (const s of stones) board[s.y][s.x] = s.color === 'B' ? 'black' : 'white';
  return board;
}

const toEngineMoves = (moves: GameMove[]): Move[] =>
  moves.map((m) => ({ x: m.x, y: m.y, player: toPlayer(m.color) }));

/** Argmax over the on-board policy (illegal moves are -1; pass is index 361). */
function policyTopMove(policy: ArrayLike<number>): { x: number; y: number } | null {
  let best = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
    if (policy[i] > bestVal) { bestVal = policy[i]; best = i; }
  }
  return best >= 0 && bestVal >= 0 ? { x: best % BOARD_SIZE, y: Math.floor(best / BOARD_SIZE) } : null;
}

// Points behind the best-SCORED candidate (mover's perspective): a common loss
// baseline across engines, so the best move is 0.0 and losses are non-negative.
// Raw engine pointsLost is relative to the order-0 move, whose score can lag
// mid-search (the negative-"better than best" display artifact).
function normalizeLosses(moves: WebCandidate[], toPlay: Color): WebCandidate[] {
  if (!moves.length) return moves;
  const side = (s: number) => (toPlay === 'B' ? s : -s);
  const best = Math.max(...moves.map((m) => side(m.scoreLead)));
  return moves.map((m) => ({ ...m, pointsLost: Math.max(0, best - side(m.scoreLead)) }));
}

function toWeb(a: KataGoAnalysisPayload, toPlay: Color): WebAnalysis {
  return {
    rootWinrate: a.rootWinRate,
    rootScoreLead: a.rootScoreLead,
    rootVisits: a.rootVisits,
    policyTop: policyTopMove(a.policy),
    moves: normalizeLosses(
      a.moves.slice(0, 15).map((m) => ({
        x: m.x, y: m.y, winrate: m.winRate, scoreLead: m.scoreLead,
        visits: m.visits, pointsLost: m.pointsLost, order: m.order,
      })),
      toPlay,
    ),
  };
}

// Where the nets are served from. VITE_NETS_BASE_URL points at a CDN holding
// the same `katago/<file>` layout (public weights, so no auth); unset falls back
// to Firebase Storage, whose allowlist gate exists only to cap egress.
const NETS_BASE = (import.meta.env.VITE_NETS_BASE_URL ?? '').replace(/\/+$/, '');

// getDownloadURL is a gated network call (Storage rules) — resolve each net once.
const urlCache = new Map<string, Promise<string>>();
function storageUrl(path: string): Promise<string> {
  if (NETS_BASE) return Promise.resolve(`${NETS_BASE}/${path}`);
  let url = urlCache.get(path);
  if (!url) { url = getDownloadURL(ref(storage, path)); urlCache.set(path, url); }
  return url;
}
const netUrl = (model: AnalysisModel): Promise<string> => storageUrl(model.netPath!);

/** Value-only score estimate (Black perspective) for every position of a game,
 * for the review score graph. Batched forward passes (browser models only);
 * `onChunk(fromMove, blackScores)` streams results so the graph fills in. */
export async function scoreTrajectory(args: {
  model: AnalysisModel;
  positions: Array<{ stones: Stone[]; previousStones?: Stone[]; previousPreviousStones?: Stone[]; moves: GameMove[]; toPlay: Color }>;
  initialStones?: Stone[];   // setup stones (handicap) — the local backend rebuilds from these + moves
  komi?: number;
  chunk?: number;
  onChunk: (fromMove: number, blackScores: number[]) => void;
  signal?: AbortSignal;
}): Promise<void> {
  // Native backend has no batch primitive, so evaluate each position with a
  // cheap low-visit search and stream results as they land (the curve fills in).
  if (args.model.kind !== 'browser') {
    for (let i = 0; i < args.positions.length; i++) {
      if (args.signal?.aborted) return;
      const p = args.positions[i];
      try {
        const lead = await valueOf(args.model, p.stones, p.toPlay, p.moves, args.komi, args.signal, args.initialStones);
        if (args.signal?.aborted) return;
        args.onChunk(i, [lead]);
      } catch {
        if (args.signal?.aborted) return;   // else skip this point and keep going
      }
    }
    return;
  }
  const client = getKataGoEngineClient();
  const modelUrl = await netUrl(args.model);
  const chunk = args.chunk ?? recommendedBatchSize();
  for (let i = 0; i < args.positions.length; i += chunk) {
    if (args.signal?.aborted) return;
    const slice = args.positions.slice(i, i + chunk);
    const evals = await client.evaluateBatch({
      modelUrl,
      positions: slice.map((p) => ({
        board: toBoardState(p.stones),
        previousBoard: p.previousStones ? toBoardState(p.previousStones) : undefined,
        previousPreviousBoard: p.previousPreviousStones ? toBoardState(p.previousPreviousStones) : undefined,
        currentPlayer: toPlayer(p.toPlay),
        moveHistory: toEngineMoves(p.moves),
        komi: args.komi ?? 7.5,
      })),
      rules: 'chinese',
    });
    if (args.signal?.aborted) return;
    args.onChunk(i, evals.map((e) => e.rootScoreLead));
  }
}

const HUMAN_NET_PATH = 'katago/b18c384nbt-humanv0.bin.gz';

/** Opponent strengths offered for human-like play — humanSL rank profiles. */
export const HUMAN_RANKS: { value: string; label: string }[] = [
  ...[18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1].map((k) => ({ value: `rank_${k}k`, label: `${k} kyu` })),
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => ({ value: `rank_${d}d`, label: `${d} dan` })),
];

export const rankLabel = (rank: string) =>
  HUMAN_RANKS.find((r) => r.value === rank)?.label ?? rank;

/** Runtimes that can drive the human net: the browser always, the native
 * backend only where it's reachable (dev). Same net either way. */
export const HUMAN_ENGINES: { id: 'browser' | 'local'; name: string; runtime: string }[] = [
  { id: 'browser', name: 'b18c384nbt-humanv0', runtime: 'WebGPU' },
  { id: 'local', name: 'b18c384nbt-humanv0', runtime: 'Metal (native)' },
];

// Profile for the score readout: the strongest the meta-encoder supports.
// Conditioning shifts the value head too, so reading the score at the play
// rank makes weak-bot score estimates (and alert-mode gaps) unreliable.
const SCORE_PROFILE = 'preaz_9d';

/** Play a human-like move: sample the human net's rank-conditioned policy (like
 * the backend /genmove). Excludes pass and the ko point; temperature < 1 sharpens.
 * The score estimate comes from a second pass conditioned on SCORE_PROFILE.
 * Returns the move (Black-perspective score estimate) or a pass when no legal move. */
export async function genmoveBrowser(args: {
  stones: Stone[];
  previousStones?: Stone[];
  moves: GameMove[];
  toPlay: Color;
  rank: string;
  temperature: number;
  komi?: number;
  koPoint?: { x: number; y: number } | null;
}): Promise<{ move: { x: number; y: number } | null; scoreLead: number }> {
  const client = getKataGoEngineClient();
  // Don't queue behind a leftover analysis search (e.g. a still-pondering
  // review) — preempt it so the human net loads and replies now.
  client.cancelAnalyses();
  const query = {
    modelUrl: await storageUrl(HUMAN_NET_PATH),
    board: toBoardState(args.stones),
    previousBoard: args.previousStones ? toBoardState(args.previousStones) : undefined,
    currentPlayer: toPlayer(args.toPlay),
    moveHistory: toEngineMoves(args.moves),
    komi: args.komi ?? 7.5,
    rules: 'chinese' as const,
  };
  const [{ policy }, scorePass] = await Promise.all([
    client.humanPolicy({ ...query, humanSLProfile: args.rank }),
    client.humanPolicy({ ...query, humanSLProfile: SCORE_PROFILE }),
  ]);
  const rootScoreLead = scorePass.rootScoreLead;

  const koIdx = args.koPoint ? args.koPoint.y * BOARD_SIZE + args.koPoint.x : -1;
  const temp = Math.max(0.05, args.temperature);
  const weights = new Float64Array(BOARD_SIZE * BOARD_SIZE);
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    if (i === koIdx || policy[i] <= 0) continue;
    const w = Math.pow(policy[i], 1 / temp);
    weights[i] = w;
    total += w;
  }
  if (total <= 0) return { move: null, scoreLead: rootScoreLead };

  let r = Math.random() * total;
  let chosen = 0;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) { chosen = i; break; }
  }
  return { move: { x: chosen % BOARD_SIZE, y: Math.floor(chosen / BOARD_SIZE) }, scoreLead: rootScoreLead };
}

export type AnalyzeArgs = {
  model: AnalysisModel;
  stones: Stone[];
  // Boards one and two plies back, so the browser engine can find the ko point
  // (and prev ladder features). Without them a just-taken ko isn't masked and
  // the immediate recapture can be suggested. The local backend replays `moves`.
  previousStones?: Stone[];
  previousPreviousStones?: Stone[];
  moves: GameMove[];
  toPlay: Color;
  positionId: string;
  // The previous position's id when navigating one ply forward — lets the
  // worker re-root its existing search tree into the child instead of starting
  // over (browser only; requires reuseTree).
  parentPositionId?: string;
  visits: number;
  komi?: number;
  // GPU dispatch batch (browser only). Omit for auto — the engine sizes it to a
  // latency budget from its on-load forward-pass measurement.
  batchSize?: number;
  signal?: AbortSignal;
  onProgress?: (partial: WebAnalysis) => void;   // browser only — mid-search updates
  // The next move to always eval, with the board after it played (for the
  // fallback value eval when the search never visited that move).
  evalNext?: { move: { x: number; y: number }; stones: Stone[] } | null;
  // Setup stones not in the move list (e.g. a tsumego's initial position); the
  // local backend rebuilds from moves, so it needs these separately. Default [].
  initialStones?: Stone[];
  // Restrict the search to a rectangle (tsumego explore hints). Both engines
  // support it — the browser via region-of-interest, the local via allow-moves.
  region?: { colMin: number; colMax: number; rowMin: number; rowMax: number } | null;
  // Run in the worker's background analysis group: preempted by interactive
  // analyses instead of superseding them (pre-warms, prefetches).
  background?: boolean;
  // Browser only. Time budget for the search (the worker defaults to 800ms —
  // an interactive-latency cap that silently truncates big visit budgets).
  maxTimeMs?: number;
  // Browser only. Keep + continue the worker's search tree across calls with
  // the same positionId (pondering); default discards it per call.
  reuseTree?: boolean;
};

export function emptyPointsIn(
  region: { colMin: number; colMax: number; rowMin: number; rowMax: number }, stones: Stone[],
): { x: number; y: number }[] {
  const occ = new Set(stones.map((s) => `${s.x},${s.y}`));
  const pts: { x: number; y: number }[] = [];
  for (let y = region.rowMin; y <= region.rowMax; y += 1) {
    for (let x = region.colMin; x <= region.colMax; x += 1) {
      if (!occ.has(`${x},${y}`)) pts.push({ x, y });
    }
  }
  return pts;
}

/** Analyze one position with the given model at `visits` playouts. Resolves to
 * null when superseded/canceled. When `evalNext` is set, the result carries a
 * `playedEval` for that move even if the search never visited it. */
export async function analyzePosition(args: AnalyzeArgs): Promise<WebAnalysis | null> {
  const analysis = args.model.kind === 'local' ? await analyzeLocal(args) : await analyzeBrowser(args);
  if (!analysis || !args.evalNext) return analysis;
  return { ...analysis, playedEval: await evalPlayedMove(args, analysis) };
}

export async function evalPlayedMove(args: AnalyzeArgs, analysis: WebAnalysis): Promise<WebAnalysis['playedEval']> {
  const nm = args.evalNext!.move;
  const cand = analysis.moves.find((m) => m.x === nm.x && m.y === nm.y);
  if (cand) return { scoreLead: cand.scoreLead, pointsLost: cand.pointsLost };
  if (args.signal?.aborted) return null;
  try {
    // The search skipped this move — evaluate the position after it directly.
    const opponent: Color = args.toPlay === 'B' ? 'W' : 'B';
    const childScore = await valueOf(
      args.model,
      args.evalNext!.stones,
      opponent,
      [...args.moves, { color: args.toPlay, x: nm.x, y: nm.y }],
      args.komi,
      args.signal,
      args.initialStones,
    );
    const side = (s: number) => (args.toPlay === 'B' ? s : -s);
    const best = analysis.moves.length ? side(analysis.moves[0].scoreLead) : side(analysis.rootScoreLead);
    return { scoreLead: childScore, pointsLost: Math.max(0, best - side(childScore)) };
  } catch {
    return null;
  }
}

/** Black-perspective score of a position (a fast value estimate). */
async function valueOf(
  model: AnalysisModel, stones: Stone[], toPlay: Color, moves: GameMove[],
  komi: number | undefined, signal?: AbortSignal, initialStones: Stone[] = [],
): Promise<number> {
  if (model.kind === 'local') {
    const a = await backendAnalyze({
      initialStones,
      moves: moves.map((m) => ({ color: m.color, x: m.x, y: m.y })),
      initialPlayer: moves[0]?.color ?? 'B',
      toPlay,
      maxVisits: 8,
      signal,
    });
    return a.root.score_lead;
  }
  const e = await getKataGoEngineClient().evaluate({
    modelUrl: await netUrl(model),
    board: toBoardState(stones),
    currentPlayer: toPlayer(toPlay),
    moveHistory: toEngineMoves(moves),
    komi: komi ?? 7.5,
    rules: 'chinese',
  });
  return e.rootScoreLead;
}

async function analyzeBrowser(args: AnalyzeArgs): Promise<WebAnalysis | null> {
  // No explicit init(): the worker loads (and caches) the model on first analyze.
  const client = getKataGoEngineClient();
  const modelUrl = await netUrl(args.model);
  try {
    const a = await client.analyze({
      analysisGroup: args.background ? 'background' : 'interactive',
      maxTimeMs: args.maxTimeMs,
      reuseTree: args.reuseTree,
      positionId: args.positionId,
      parentPositionId: args.parentPositionId,
      modelUrl,
      board: toBoardState(args.stones),
      previousBoard: args.previousStones ? toBoardState(args.previousStones) : undefined,
      previousPreviousBoard: args.previousPreviousStones ? toBoardState(args.previousPreviousStones) : undefined,
      currentPlayer: toPlayer(args.toPlay),
      moveHistory: toEngineMoves(args.moves),
      komi: args.komi ?? 7.5,
      rules: 'chinese',
      visits: args.visits,
      batchSize: args.batchSize,
      topK: 15,
      regionOfInterest: args.region
        ? { xMin: args.region.colMin, xMax: args.region.colMax, yMin: args.region.rowMin, yMax: args.region.rowMax }
        : null,
      reportDuringSearchEveryMs: args.onProgress ? 120 : undefined,
      onProgress: args.onProgress ? (p) => args.onProgress!(toWeb(p, args.toPlay)) : undefined,
    });
    return toWeb(a, args.toPlay);
  } catch (err) {
    if (isKataGoCanceledError(err)) return null;
    throw err;
  }
}

export function mapBackend(a: BackendAnalysis, toPlay: Color): WebAnalysis {
  // Same loss baseline as toWeb: points behind the best-scored candidate.
  const sideValue = (lead: number) => (toPlay === 'B' ? lead : -lead);
  const best = a.moves.length ? Math.max(...a.moves.map((m) => sideValue(m.score_lead))) : 0;
  // policyTop = the highest-prior legal candidate (the net's raw pick).
  let policyTop: { x: number; y: number } | null = null;
  let bestPrior = -Infinity;
  for (const m of a.moves) {
    if (m.x != null && m.y != null && m.prior > bestPrior) { bestPrior = m.prior; policyTop = { x: m.x, y: m.y }; }
  }
  return {
    rootWinrate: a.root.winrate,
    rootScoreLead: a.root.score_lead,
    rootVisits: a.root.visits,
    policyTop,
    moves: a.moves
      .filter((m) => m.x != null && m.y != null)
      .slice(0, 15)
      .map((m) => ({
        x: m.x as number, y: m.y as number,
        winrate: m.winrate, scoreLead: m.score_lead, visits: m.visits,
        pointsLost: best - sideValue(m.score_lead), order: m.order,
      })),
  };
}

async function analyzeLocal(args: AnalyzeArgs): Promise<WebAnalysis | null> {
  const base: Omit<AnalyzeParams, 'maxVisits'> = {
    initialStones: (args.initialStones ?? []).map((s) => ({ x: s.x, y: s.y, color: s.color })),
    moves: args.moves.map((m) => ({ color: m.color, x: m.x, y: m.y })),
    initialPlayer: args.moves[0]?.color ?? 'B',
    toPlay: args.toPlay,
    allowMoves: args.region ? emptyPointsIn(args.region, args.stones) : undefined,
    signal: args.signal,
  };
  try {
    // The backend doesn't stream, so run a fast low-visit pass first to show the
    // top move + spinner while the full search runs. (1 visit returns no
    // candidates; 2 is the minimum that does, and gives the same policy move.)
    if (args.onProgress && args.visits > 2) {
      const quick = await backendAnalyze({ ...base, maxVisits: 2 });
      if (!args.signal?.aborted) args.onProgress(mapBackend(quick, args.toPlay));
    }
    if (args.signal?.aborted) return null;
    return mapBackend(await backendAnalyze({ ...base, maxVisits: args.visits }), args.toPlay);
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return null;
    throw err;
  }
}
