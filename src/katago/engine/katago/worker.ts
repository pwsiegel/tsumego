/// <reference lib="webworker" />

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import pako from 'pako';

import type { KataGoAnalyzeRequest, KataGoWorkerRequest, KataGoWorkerResponse } from './types';
import type { BoardState, GameRules, Move, Player, RegionOfInterest } from '../../types';
import { getAnimationNow } from '../../utils/animationFrame';
import { parseKataGoModelV8 } from './loadModelV8';
import { KataGoModelV8Tf } from './modelV8';
import { ENGINE_MAX_TIME_MS, ENGINE_MAX_VISITS } from './limits';
import { autoBatchSize, type EnginePerf } from './autoBatch';
import { MctsSearch, type OwnershipMode } from './analyzeMcts';
import { fillInputsV7Fast, type RecentMove } from './featuresV7Fast';
import {
  BLACK,
  BOARD_AREA,
  BOARD_SIZE,
  PASS_MOVE,
  WHITE,
  computeAreaMapV7KataGoInto,
  computeLadderFeaturesV7KataGoInto,
  computeLadderedStonesV7KataGoInto,
  computeLibertyMapInto,
  playMove,
  setBoardSize,
  type SimPosition,
  type StoneColor,
} from './fastBoard';
import { postprocessKataGoV8 } from './evalV8';
import { fillPositionInputsV7 } from './positionFeatures';
import { buildSGFMetadataV1 } from './sgfMetadata';

let model: KataGoModelV8Tf | null = null;
let loadedModelName: string | undefined;
let loadedModelUrl: string | null = null;
// Per-net, per-device forward-pass timings, measured once when the net loads.
// Drives auto batch sizing when a request omits an explicit batchSize.
let enginePerf: EnginePerf | null = null;
let backendPromise: Promise<void> | null = null;
let prodModeEnabled = false;
let queue: Promise<void> = Promise.resolve();

let V7_SPATIAL_STRIDE = BOARD_AREA * 22;
const V7_GLOBAL_STRIDE = 19;

let evalSpatialV7 = new Float32Array(V7_SPATIAL_STRIDE);
let evalGlobalV7 = new Float32Array(V7_GLOBAL_STRIDE);

let stonesScratch = new Uint8Array(BOARD_AREA);
let prevStonesScratch = new Uint8Array(BOARD_AREA);
let prevPrevStonesScratch = new Uint8Array(BOARD_AREA);

let koSimStonesScratch = new Uint8Array(BOARD_AREA);
let koSimPosScratch: SimPosition = { stones: koSimStonesScratch, koPoint: -1 };
const koCaptureStackScratch: number[] = [];

let libertyMapScratch = new Uint8Array(BOARD_AREA);
let areaMapScratch = new Uint8Array(BOARD_AREA);

let ladderedStonesScratch = new Uint8Array(BOARD_AREA);
let ladderWorkingMovesScratch = new Uint8Array(BOARD_AREA);
let prevLadderedStonesScratch = new Uint8Array(BOARD_AREA);
let prevPrevLadderedStonesScratch = new Uint8Array(BOARD_AREA);

let evalBatchCapacity = 0;
let evalBatchSpatialV7 = new Float32Array(0);
let evalBatchGlobalV7 = new Float32Array(0);
let scratchBoardSize = BOARD_SIZE;
type ParsedKataGoModelV8 = ReturnType<typeof parseKataGoModelV8>;

function regionKey(roi?: RegionOfInterest | null): string | null {
  if (!roi) return null;
  const xMin = Math.max(0, Math.min(BOARD_SIZE - 1, Math.min(roi.xMin, roi.xMax)));
  const xMax = Math.max(0, Math.min(BOARD_SIZE - 1, Math.max(roi.xMin, roi.xMax)));
  const yMin = Math.max(0, Math.min(BOARD_SIZE - 1, Math.min(roi.yMin, roi.yMax)));
  const yMax = Math.max(0, Math.min(BOARD_SIZE - 1, Math.max(roi.yMin, roi.yMax)));
  const isSinglePoint = xMin === xMax && yMin === yMax;
  const isWholeBoard = xMin === 0 && yMin === 0 && xMax === BOARD_SIZE - 1 && yMax === BOARD_SIZE - 1;
  if (isSinglePoint || isWholeBoard) return null;
  return `${xMin},${xMax},${yMin},${yMax}`;
}

function getEvalBatchBuffersV7(batch: number): { spatial: Float32Array; global: Float32Array } {
  if (batch > evalBatchCapacity) {
    evalBatchCapacity = batch;
    evalBatchSpatialV7 = new Float32Array(batch * V7_SPATIAL_STRIDE);
    evalBatchGlobalV7 = new Float32Array(batch * V7_GLOBAL_STRIDE);
  }
  return {
    spatial: evalBatchSpatialV7.subarray(0, batch * V7_SPATIAL_STRIDE),
    global: evalBatchGlobalV7.subarray(0, batch * V7_GLOBAL_STRIDE),
  };
}

function playerToColor(p: Player): StoneColor {
  return p === 'black' ? BLACK : WHITE;
}

function boardStateToStonesInto(board: BoardState, out: Uint8Array): void {
  out.fill(0);
  for (let y = 0; y < BOARD_SIZE; y++) {
    const row = board[y];
    for (let x = 0; x < BOARD_SIZE; x++) {
      const v = row?.[x] ?? null;
      if (!v) continue;
      out[y * BOARD_SIZE + x] = v === 'black' ? BLACK : WHITE;
    }
  }
}

function movesToRecentMoves(moves: Move[]): RecentMove[] {
  const out = new Array<RecentMove>(moves.length);
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i]!;
    out[i] = {
      move: m.x < 0 || m.y < 0 ? PASS_MOVE : m.y * BOARD_SIZE + m.x,
      player: m.player,
    };
  }
  return out;
}

function countHistoryTurnsIncluded(args: { recentMoves: RecentMove[]; currentPlayer: Player; conservativePassAndIsRoot: boolean }): number {
  const lastMove = args.recentMoves.length > 0 ? args.recentMoves[args.recentMoves.length - 1] : null;
  const passWouldEndGame = lastMove?.move === PASS_MOVE;
  if (args.conservativePassAndIsRoot && passWouldEndGame) return 0;

  const pla = args.currentPlayer;
  const opp = pla === 'black' ? 'white' : 'black';
  const expectedPlayers: Player[] = [opp, pla, opp, pla, opp];

  let included = 0;
  for (let i = 0; i < 5; i++) {
    const m = args.recentMoves[args.recentMoves.length - 1 - i];
    if (!m) break;
    if (m.player !== expectedPlayers[i]) break;
    included++;
  }
  return included;
}

function computeKoPointAfterMove(previousStones: Uint8Array, move: Move | null): number {
  if (!move || move.x < 0 || move.y < 0) return -1;

  koSimStonesScratch.set(previousStones);
  koSimPosScratch.koPoint = -1;
  koCaptureStackScratch.length = 0;

  try {
    playMove(koSimPosScratch, move.y * BOARD_SIZE + move.x, playerToColor(move.player), koCaptureStackScratch);
    return koSimPosScratch.koPoint;
  } catch {
    return -1;
  }
}

function fillInputsV7FastForPosition(args: {
  board: BoardState;
  previousBoard?: BoardState;
  previousPreviousBoard?: BoardState;
  currentPlayer: Player;
  moveHistory: Move[];
  komi: number;
  rules: GameRules;
  conservativePassAndIsRoot: boolean;
  outSpatial: Float32Array;
  outGlobal: Float32Array;
}): void {
  boardStateToStonesInto(args.board, stonesScratch);

  if (args.previousBoard) boardStateToStonesInto(args.previousBoard, prevStonesScratch);
  else prevStonesScratch.set(stonesScratch);

  if (args.previousPreviousBoard) boardStateToStonesInto(args.previousPreviousBoard, prevPrevStonesScratch);
  else prevPrevStonesScratch.set(prevStonesScratch);

  const lastMove = args.moveHistory.length > 0 ? args.moveHistory[args.moveHistory.length - 1]! : null;
  const prevMove = args.moveHistory.length >= 2 ? args.moveHistory[args.moveHistory.length - 2]! : null;

  const koPoint = args.previousBoard ? computeKoPointAfterMove(prevStonesScratch, lastMove) : -1;
  const prevKoPoint = args.previousPreviousBoard ? computeKoPointAfterMove(prevPrevStonesScratch, prevMove) : -1;
  const prevPrevKoPoint = -1;

  const recentMoves = movesToRecentMoves(args.moveHistory);
  const numTurnsOfHistoryIncluded = countHistoryTurnsIncluded({
    recentMoves,
    currentPlayer: args.currentPlayer,
    conservativePassAndIsRoot: args.conservativePassAndIsRoot,
  });

  const prevLadderStones = numTurnsOfHistoryIncluded < 1 ? stonesScratch : prevStonesScratch;
  const prevLadderKoPoint = numTurnsOfHistoryIncluded < 1 ? koPoint : prevKoPoint;

  const prevPrevLadderStones = numTurnsOfHistoryIncluded < 2 ? prevLadderStones : prevPrevStonesScratch;
  const prevPrevLadderKoPoint = numTurnsOfHistoryIncluded < 2 ? prevLadderKoPoint : prevPrevKoPoint;

  computeLibertyMapInto(stonesScratch, libertyMapScratch);
  if (args.rules === 'chinese') computeAreaMapV7KataGoInto(stonesScratch, areaMapScratch);

  computeLadderFeaturesV7KataGoInto({
    stones: stonesScratch,
    koPoint,
    currentPlayer: playerToColor(args.currentPlayer),
    outLadderedStones: ladderedStonesScratch,
    outLadderWorkingMoves: ladderWorkingMovesScratch,
  });
  computeLadderedStonesV7KataGoInto({
    stones: prevLadderStones,
    koPoint: prevLadderKoPoint,
    outLadderedStones: prevLadderedStonesScratch,
  });
  computeLadderedStonesV7KataGoInto({
    stones: prevPrevLadderStones,
    koPoint: prevPrevLadderKoPoint,
    outLadderedStones: prevPrevLadderedStonesScratch,
  });

  fillInputsV7Fast({
    stones: stonesScratch,
    koPoint,
    currentPlayer: args.currentPlayer,
    recentMoves,
    komi: args.komi,
    rules: args.rules,
    conservativePassAndIsRoot: args.conservativePassAndIsRoot,
    libertyMap: libertyMapScratch,
    areaMap: args.rules === 'chinese' ? areaMapScratch : undefined,
    ladderedStones: ladderedStonesScratch,
    prevLadderedStones: prevLadderedStonesScratch,
    prevPrevLadderedStones: prevPrevLadderedStonesScratch,
    ladderWorkingMoves: ladderWorkingMovesScratch,
    outSpatial: args.outSpatial,
    outGlobal: args.outGlobal,
  });
}

let search: MctsSearch | null = null;
let searchKey: {
  positionId: string;
  positionKey: string | null;
  modelUrl: string;
  boardSize: number;
  maxChildren: number;
  ownershipMode: OwnershipMode;
  komi: number;
  currentPlayer: 'black' | 'white';
  wideRootNoise: number;
  rootSymmetrySamples: number;
  rules: GameRules;
  nnRandomize: boolean;
  conservativePass: boolean;
  roiKey: string | null;
} | null = null;
const latestAnalyzeByGroup = new Map<string, number>();
let interactiveToken = 0;
const analyzeMeta = new WeakMap<KataGoAnalyzeRequest, { analysisGroup: 'interactive' | 'background'; interactiveToken: number }>();

function ensureBoardSizeForWorker(boardSize: number): void {
  if (boardSize === scratchBoardSize) return;
  setBoardSize(boardSize);
  scratchBoardSize = BOARD_SIZE;
  V7_SPATIAL_STRIDE = BOARD_AREA * 22;
  evalSpatialV7 = new Float32Array(V7_SPATIAL_STRIDE);
  evalGlobalV7 = new Float32Array(V7_GLOBAL_STRIDE);
  stonesScratch = new Uint8Array(BOARD_AREA);
  prevStonesScratch = new Uint8Array(BOARD_AREA);
  prevPrevStonesScratch = new Uint8Array(BOARD_AREA);
  koSimStonesScratch = new Uint8Array(BOARD_AREA);
  koSimPosScratch = { stones: koSimStonesScratch, koPoint: -1 };
  libertyMapScratch = new Uint8Array(BOARD_AREA);
  areaMapScratch = new Uint8Array(BOARD_AREA);
  ladderedStonesScratch = new Uint8Array(BOARD_AREA);
  ladderWorkingMovesScratch = new Uint8Array(BOARD_AREA);
  prevLadderedStonesScratch = new Uint8Array(BOARD_AREA);
  prevPrevLadderedStonesScratch = new Uint8Array(BOARD_AREA);
  evalBatchCapacity = 0;
  evalBatchSpatialV7 = new Float32Array(0);
  evalBatchGlobalV7 = new Float32Array(0);
  search = null;
  searchKey = null;
}

// WebGPU is the only backend we run. The CPU/wasm paths reached the same
// answers roughly 12x slower per evaluation, which cannot finish a useful
// search, so failing loudly beats degrading into something unusable.
const NO_WEBGPU =
  'WebGPU is unavailable in this browser, and the analysis net runs on WebGPU only.';

async function initBackend(): Promise<void> {
  try {
    await tf.setBackend('webgpu');
    await tf.ready();
  } catch (err) {
    throw new Error(`${NO_WEBGPU} (${err instanceof Error ? err.message : String(err)})`);
  }
  if (tf.getBackend() !== 'webgpu') throw new Error(NO_WEBGPU);
}

function maybeUngzip(data: Uint8Array): Uint8Array {
  // gzip magic bytes 0x1f8b
  if (data.length >= 2 && data[0] === 0x1f && data[1] === 0x8b) return pako.ungzip(data);
  return data;
}

async function ensureBackend(): Promise<void> {
  if (backendPromise) {
    await backendPromise;
    return;
  }

  model?.dispose();
  model = null;
  loadedModelName = undefined;
  loadedModelUrl = null;
  enginePerf = null;
  search = null;
  searchKey = null;

  backendPromise = initBackend()
      .then(() => {
        if (!prodModeEnabled) {
          tf.enableProdMode();
          prodModeEnabled = true;
        }
      })
      .catch((err) => {
        backendPromise = null;
        throw err;
      });
  await backendPromise;
}

async function warmupModel(candidate: KataGoModelV8Tf): Promise<void> {
  const spatial = tf.zeros([1, 19, 19, 22], 'float32') as tf.Tensor4D;
  const global = tf.zeros([1, 19], 'float32') as tf.Tensor2D;
  let out: ReturnType<KataGoModelV8Tf['forwardValueOnly']> | null = null;
  try {
    out = candidate.forwardValueOnly(spatial, global);
    const results = await Promise.allSettled([out.value.data(), out.scoreValue.data()]);
    for (const result of results) {
      if (result.status === 'rejected') throw result.reason;
    }
  } finally {
    spatial.dispose();
    global.dispose();
    out?.value.dispose();
    out?.scoreValue.dispose();
  }
}

// Time a value-only forward pass at two batch sizes (steady state — the warmup
// above already compiled the base pipelines) so the app can size dispatches to
// a latency budget. Each size gets one discarded warm run (its pipeline may not
// be compiled yet), then the fastest of two timed runs (min = least noisy).
async function measureEnginePerf(m: KataGoModelV8Tf): Promise<EnginePerf> {
  const points: { batch: number; ms: number }[] = [];
  for (const batch of [2, 12]) {
    const spatial = tf.zeros([batch, 19, 19, 22], 'float32') as tf.Tensor4D;
    const global = tf.zeros([batch, 19], 'float32') as tf.Tensor2D;
    try {
      const run = async () => {
        const out = m.forwardValueOnly(spatial, global);
        await Promise.all([out.value.data(), out.scoreValue.data()]);
        out.value.dispose();
        out.scoreValue.dispose();
      };
      await run();
      let best = Infinity;
      for (let i = 0; i < 2; i++) {
        const t = getAnimationNow();
        await run();
        best = Math.min(best, getAnimationNow() - t);
      }
      points.push({ batch, ms: best });
    } finally {
      spatial.dispose();
      global.dispose();
    }
  }
  return { points };
}

async function createWarmedModel(parsed: ParsedKataGoModelV8): Promise<KataGoModelV8Tf> {
  const candidate = new KataGoModelV8Tf(parsed);
  try {
    await warmupModel(candidate);
    return candidate;
  } catch (err) {
    candidate.dispose();
    throw err;
  }
}

function installModel(nextModel: KataGoModelV8Tf, parsed: ParsedKataGoModelV8, modelUrl: string): void {
  model?.dispose();
  model = nextModel;
  loadedModelName = parsed.modelName;
  loadedModelUrl = modelUrl;
  enginePerf = null;
  search = null;
  searchKey = null;
}

// Human-readable net name from a (Firebase Storage) model URL, for status
// events before the file is parsed: ".../o/katago%2Fkata1-b18.bin.gz?token=…".
function modelNameFromUrl(modelUrl: string): string {
  try {
    const path = decodeURIComponent(new URL(modelUrl).pathname);
    return path.split('/').pop()!.replace(/\.bin\.gz$/, '');
  } catch {
    return 'model';
  }
}

async function ensureModel(modelUrl: string): Promise<void> {
  try {
    await ensureModelInner(modelUrl);
  } catch (err) {
    post({
      type: 'katago:model_status', status: 'error',
      modelName: modelNameFromUrl(modelUrl),
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
}

async function ensureModelInner(modelUrl: string): Promise<void> {
  await ensureBackend();
  if (model && loadedModelUrl === modelUrl) return;

  post({ type: 'katago:model_status', status: 'loading', modelName: modelNameFromUrl(modelUrl) });
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`Failed to fetch model: ${res.status} ${res.statusText}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const data = maybeUngzip(buf);

  const parsed = parseKataGoModelV8(data);
  installModel(await createWarmedModel(parsed), parsed, modelUrl);
  if (model) enginePerf = await measureEnginePerf(model);
  post({ type: 'katago:model_status', status: 'ready', modelName: loadedModelName ?? parsed.modelName });
}

function post(msg: KataGoWorkerResponse, transfer?: Transferable[]) {
  if (transfer && transfer.length > 0) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

async function handleMessage(msg: KataGoWorkerRequest): Promise<void> {
  if (msg.type === 'katago:init') {
    await ensureModel(msg.modelUrl);
    post({
      type: 'katago:init_result',
      ok: true,
      backend: tf.getBackend(),
      modelName: loadedModelName,
      perf: enginePerf ?? undefined,
    });
    return;
  }

  if (msg.type === 'katago:eval') {
    await ensureModel(msg.modelUrl);
    if (!model) throw new Error('Model not loaded');
    ensureBoardSizeForWorker(msg.board.length);
    const boardSize = BOARD_SIZE;

    const conservativePass = msg.conservativePass !== false;
    const rules: GameRules = msg.rules === 'chinese' ? 'chinese' : msg.rules === 'korean' ? 'korean' : 'japanese';

    fillInputsV7FastForPosition({
      board: msg.board,
      previousBoard: msg.previousBoard,
      previousPreviousBoard: msg.previousPreviousBoard,
      currentPlayer: msg.currentPlayer,
      moveHistory: msg.moveHistory,
      komi: msg.komi,
      rules,
      conservativePassAndIsRoot: conservativePass,
      outSpatial: evalSpatialV7,
      outGlobal: evalGlobalV7,
    });

    const spatial = tf.tensor4d(evalSpatialV7, [1, boardSize, boardSize, 22]);
    const global = tf.tensor2d(evalGlobalV7, [1, 19]);
    const out = model.forwardValueOnly(spatial, global);
    const [valueLogitsArr, scoreValueArr] = await Promise.all([out.value.data(), out.scoreValue.data()]);
    spatial.dispose();
    global.dispose();
    out.value.dispose();
    out.scoreValue.dispose();

    const evaled = postprocessKataGoV8({
      nextPlayer: msg.currentPlayer,
      valueLogits: valueLogitsArr,
      scoreValue: scoreValueArr,
      postProcessParams: model.postProcessParams,
    });

    post({
      type: 'katago:eval_result',
      id: msg.id,
      ok: true,
      backend: tf.getBackend(),
      modelName: loadedModelName,
      perf: enginePerf ?? undefined,
      eval: {
        rootWinRate: evaled.blackWinProb,
        rootScoreLead: evaled.blackScoreLead,
        rootScoreSelfplay: evaled.blackScoreMean,
        rootScoreStdev: evaled.blackScoreStdev,
      },
    });
    return;
  }

  if (msg.type === 'katago:human_policy') {
    await ensureModel(msg.modelUrl);
    if (!model) throw new Error('Model not loaded');
    if (!model.hasMetaEncoder) throw new Error('Model has no metadata encoder — not a human-SL net');
    ensureBoardSizeForWorker(msg.board.length);
    const boardSize = BOARD_SIZE;
    const rules: GameRules = msg.rules === 'chinese' ? 'chinese' : msg.rules === 'korean' ? 'korean' : 'japanese';

    // KataGo evaluates the human net's root with pre-root history ignored
    // (analysis-engine default) — feeding move history diverges from native.
    fillPositionInputsV7({
      board: msg.board,
      previousBoard: msg.previousBoard,
      previousPreviousBoard: msg.previousPreviousBoard,
      currentPlayer: msg.currentPlayer,
      moveHistory: msg.moveHistory,
      komi: msg.komi,
      rules,
      conservativePassAndIsRoot: true,
      maxHistory: 0,
      outSpatial: evalSpatialV7,
      outGlobal: evalGlobalV7,
    });

    const spatial = tf.tensor4d(evalSpatialV7, [1, boardSize, boardSize, 22]);
    const global = tf.tensor2d(evalGlobalV7, [1, 19]);
    const meta = tf.tensor2d(buildSGFMetadataV1(msg.humanSLProfile), [1, 192]);
    const out = model.forwardPolicyValue(spatial, global, meta);
    const [boardLogits, passArr, valueArr, scoreArr] = await Promise.all([
      out.policy.slice([0, 0, 0, 0], [1, boardSize, boardSize, 1]).reshape([boardSize * boardSize]).data(),
      out.policyPass.slice([0, 0], [1, 1]).reshape([1]).data(),
      out.value.data(),
      out.scoreValue.data(),
    ]);
    const passLogit = passArr[0]!;
    spatial.dispose();
    global.dispose();
    meta.dispose();
    out.policy.dispose();
    out.policyPass.dispose();
    out.value.dispose();
    out.scoreValue.dispose();

    const evaled = postprocessKataGoV8({
      nextPlayer: msg.currentPlayer,
      valueLogits: valueArr,
      scoreValue: scoreArr,
      postProcessParams: model.postProcessParams,
    });

    // Softmax over legal (empty) board points + pass; occupied points masked out.
    const n = boardSize * boardSize;
    const policy = new Float32Array(n + 1);
    let maxL = passLogit;
    for (let i = 0; i < n; i++) {
      const x = i % boardSize;
      const y = (i / boardSize) | 0;
      if (msg.board[y]?.[x]) { policy[i] = -Infinity; continue; }
      policy[i] = boardLogits[i]!;
      if (policy[i] > maxL) maxL = policy[i];
    }
    policy[n] = passLogit;
    let sum = 0;
    for (let i = 0; i <= n; i++) { const e = Math.exp(policy[i] - maxL); policy[i] = e; sum += e; }
    for (let i = 0; i <= n; i++) policy[i] /= sum;

    post(
      {
        type: 'katago:human_policy_result',
        id: msg.id,
        ok: true,
        backend: tf.getBackend(),
        modelName: loadedModelName,
        policy,
        rootScoreLead: evaled.blackScoreLead,
      },
      [policy.buffer],
    );
    return;
  }

  if (msg.type === 'katago:eval_batch') {
    await ensureModel(msg.modelUrl);
    if (!model) throw new Error('Model not loaded');

    const conservativePass = msg.conservativePass !== false;
    const rules: GameRules = msg.rules === 'chinese' ? 'chinese' : msg.rules === 'korean' ? 'korean' : 'japanese';

    const batch = msg.positions.length;
    if (batch <= 0) {
      post({
        type: 'katago:eval_batch_result',
        id: msg.id,
        ok: true,
        backend: tf.getBackend(),
        modelName: loadedModelName,
        perf: enginePerf ?? undefined,
        evals: [],
      });
      return;
    }

    const boardSize = msg.positions[0] ? msg.positions[0].board.length : BOARD_SIZE;
    ensureBoardSizeForWorker(boardSize);
    const size = BOARD_SIZE;

    const { spatial: spatialBatch, global: globalBatch } = getEvalBatchBuffersV7(batch);

    for (let i = 0; i < batch; i++) {
      const pos = msg.positions[i]!;
      fillInputsV7FastForPosition({
        board: pos.board,
        previousBoard: pos.previousBoard,
        previousPreviousBoard: pos.previousPreviousBoard,
        currentPlayer: pos.currentPlayer,
        moveHistory: pos.moveHistory,
        komi: pos.komi,
        rules,
        conservativePassAndIsRoot: conservativePass,
        outSpatial: spatialBatch.subarray(i * V7_SPATIAL_STRIDE, (i + 1) * V7_SPATIAL_STRIDE),
        outGlobal: globalBatch.subarray(i * V7_GLOBAL_STRIDE, (i + 1) * V7_GLOBAL_STRIDE),
      });
    }

    const spatial = tf.tensor4d(spatialBatch, [batch, size, size, 22]);
    const global = tf.tensor2d(globalBatch, [batch, 19]);
    const out = model.forwardValueOnly(spatial, global);
    const [valueLogitsArr, scoreValueArr] = await Promise.all([out.value.data(), out.scoreValue.data()]);
    spatial.dispose();
    global.dispose();
    out.value.dispose();
    out.scoreValue.dispose();

    const evals = new Array(batch);
    for (let i = 0; i < batch; i++) {
      const evaled = postprocessKataGoV8({
        nextPlayer: msg.positions[i]!.currentPlayer,
        valueLogits: valueLogitsArr.subarray(i * 3, i * 3 + 3),
        scoreValue: scoreValueArr.subarray(i * 4, i * 4 + 4),
        postProcessParams: model.postProcessParams,
      });
      evals[i] = {
        rootWinRate: evaled.blackWinProb,
        rootScoreLead: evaled.blackScoreLead,
        rootScoreSelfplay: evaled.blackScoreMean,
        rootScoreStdev: evaled.blackScoreStdev,
      };
    }

    post({
      type: 'katago:eval_batch_result',
      id: msg.id,
      ok: true,
      backend: tf.getBackend(),
      modelName: loadedModelName,
      perf: enginePerf ?? undefined,
      evals,
    });
    return;
  }

  if (msg.type === 'katago:analyze') {
    const meta = analyzeMeta.get(msg);
    const analysisGroup = meta?.analysisGroup ?? msg.analysisGroup ?? 'background';
    const interactiveTokenAtEnqueue = meta?.interactiveToken ?? interactiveToken;
    const isStale = () => latestAnalyzeByGroup.get(analysisGroup) !== msg.id;
    const isPreemptedByInteractive =
      analysisGroup !== 'interactive' && interactiveToken !== interactiveTokenAtEnqueue;
    const shouldAbort = () => isStale() || isPreemptedByInteractive;
    const postCanceled = () =>
      post({
        type: 'katago:analyze_result',
        id: msg.id,
        ok: false,
        canceled: true,
        error: 'canceled',
      });

    if (shouldAbort()) {
      postCanceled();
      return;
    }

    await ensureModel(msg.modelUrl);
    if (!model) throw new Error('Model not loaded');
    if (shouldAbort()) {
      postCanceled();
      return;
    }

    ensureBoardSizeForWorker(msg.board.length);
    const boardSize = BOARD_SIZE;

    const maxVisits = Math.max(16, Math.min(msg.visits ?? 256, ENGINE_MAX_VISITS));
    const maxTimeMs = Math.max(25, Math.min(msg.maxTimeMs ?? 800, ENGINE_MAX_TIME_MS));
    // Omitted batchSize = auto: size each GPU dispatch to a latency budget from
    // the measured forward-pass timings.
    const defaultBatch = tf.getBackend() === 'webgpu' ? autoBatchSize(enginePerf) : 4;
    const batchSize = Math.max(1, Math.min(msg.batchSize ?? defaultBatch, 64));
    const maxChildren = Math.max(4, Math.min(msg.maxChildren ?? 64, BOARD_AREA));
    const topK = Math.max(1, Math.min(msg.topK ?? 10, 50));
    const includeMovesOwnership = msg.includeMovesOwnership === true;
    const requestedOwnershipMode: OwnershipMode = msg.ownershipMode ?? 'root';
    const ownershipMode: OwnershipMode = includeMovesOwnership ? 'tree' : requestedOwnershipMode;
    const analysisPvLen = Math.max(0, Math.min(msg.analysisPvLen ?? 15, 60));
    const wideRootNoise = Math.max(0, Math.min(msg.wideRootNoise ?? 0.04, 5));
    const rules: GameRules = msg.rules === 'chinese' ? 'chinese' : msg.rules === 'korean' ? 'korean' : 'japanese';
    const nnRandomize = msg.nnRandomize !== false;
    const rootSymmetrySamples = tf.getBackend() === 'webgpu' && nnRandomize ? 8 : 1;
    const conservativePass = msg.conservativePass !== false;
    const roiKey = regionKey(msg.regionOfInterest);
    const reportEveryMsRaw = msg.reportDuringSearchEveryMs;
    const reportEveryMs =
      typeof reportEveryMsRaw === 'number' && Number.isFinite(reportEveryMsRaw)
        ? Math.max(0, reportEveryMsRaw)
        : 0;
    const shouldReport = reportEveryMs > 0;
    const cloneBuffers = msg.reuseTree === true || shouldReport;

    const canReuse =
      msg.reuseTree === true &&
      typeof msg.positionId === 'string' &&
      !!search &&
      !!searchKey &&
      searchKey.positionId === msg.positionId &&
      searchKey.positionKey === (msg.positionKey ?? null) &&
      searchKey.modelUrl === msg.modelUrl &&
      searchKey.boardSize === boardSize &&
      searchKey.maxChildren === maxChildren &&
      searchKey.ownershipMode === ownershipMode &&
      searchKey.komi === msg.komi &&
      searchKey.currentPlayer === msg.currentPlayer &&
      searchKey.wideRootNoise === wideRootNoise &&
      searchKey.rootSymmetrySamples === rootSymmetrySamples &&
      searchKey.rules === rules &&
      searchKey.nnRandomize === nnRandomize &&
      searchKey.conservativePass === conservativePass &&
      searchKey.roiKey === roiKey;

    let reusedSearch = canReuse;

    // Re-root the existing search when the new position is a direct child of the current root.
    if (
      !reusedSearch &&
      msg.reuseTree === true &&
      search &&
      searchKey &&
      typeof msg.positionId === 'string' &&
      typeof msg.parentPositionId === 'string'
    ) {
      const canReRoot =
        searchKey.positionId === msg.parentPositionId &&
        searchKey.positionKey === (msg.parentPositionKey ?? null) &&
        searchKey.modelUrl === msg.modelUrl &&
        searchKey.maxChildren === maxChildren &&
        searchKey.ownershipMode === ownershipMode &&
        searchKey.komi === msg.komi &&
        searchKey.wideRootNoise === wideRootNoise &&
        searchKey.rootSymmetrySamples === rootSymmetrySamples &&
        searchKey.rules === rules &&
        searchKey.nnRandomize === nnRandomize &&
        searchKey.conservativePass === conservativePass &&
        searchKey.roiKey === roiKey;

      if (canReRoot) {
        const lastMove = msg.moveHistory[msg.moveHistory.length - 1] ?? null;
        const move =
          lastMove && lastMove.x >= 0 && lastMove.y >= 0 ? lastMove.y * BOARD_SIZE + lastMove.x : PASS_MOVE;
        if (lastMove) {
          const reRooted = await search.reRootToChild({
            move,
            board: msg.board,
            previousBoard: msg.previousBoard,
            previousPreviousBoard: msg.previousPreviousBoard,
            currentPlayer: msg.currentPlayer,
            moveHistory: msg.moveHistory,
            komi: msg.komi,
            rules,
            regionOfInterest: msg.regionOfInterest,
          });
          if (reRooted) {
            reusedSearch = true;
            searchKey = {
              positionId: msg.positionId,
              positionKey: msg.positionKey ?? null,
              modelUrl: msg.modelUrl,
              boardSize,
              maxChildren,
              ownershipMode,
              komi: msg.komi,
              currentPlayer: msg.currentPlayer,
              wideRootNoise,
              rootSymmetrySamples,
              rules,
              nnRandomize,
              conservativePass,
              roiKey,
            };
          }
        }
      }
    }

    if (!reusedSearch) {
      search = await MctsSearch.create({
        model,
        board: msg.board,
        previousBoard: msg.previousBoard,
        previousPreviousBoard: msg.previousPreviousBoard,
        currentPlayer: msg.currentPlayer,
        moveHistory: msg.moveHistory,
        komi: msg.komi,
        rules,
        nnRandomize,
        conservativePass,
        maxChildren,
        ownershipMode,
        wideRootNoise,
        rootSymmetrySamples,
        regionOfInterest: msg.regionOfInterest,
      });
      if (typeof msg.positionId === 'string') {
        searchKey = {
          positionId: msg.positionId,
          positionKey: msg.positionKey ?? null,
          modelUrl: msg.modelUrl,
          boardSize,
          maxChildren,
          ownershipMode,
          komi: msg.komi,
          currentPlayer: msg.currentPlayer,
          wideRootNoise,
          rootSymmetrySamples,
          rules,
          nnRandomize,
          conservativePass,
          roiKey,
        };
      } else {
        searchKey = null;
      }
    }

    const postAnalysis = (analysis: ReturnType<MctsSearch['getAnalysis']>, type: 'katago:analyze_update' | 'katago:analyze_result') => {
      const transfer: Transferable[] = [];
      const push = (value?: unknown) => {
        if (value && ArrayBuffer.isView(value)) transfer.push(value.buffer);
      };
      push(analysis.ownership);
      push(analysis.ownershipStdev);
      push(analysis.policy);
      for (const move of analysis.moves) push(move.ownership);

      post(
        {
          type,
          id: msg.id,
          ok: true,
          backend: tf.getBackend(),
          modelName: loadedModelName,
          analysis,
          perf: enginePerf ?? undefined,
          chosenBatchSize: batchSize,
        },
        transfer
      );
    };

    const buildAnalysis = () =>
      search!.getAnalysis({
        topK,
        includeMovesOwnership,
        analysisPvLen,
        cloneBuffers,
        ownershipRefreshIntervalMs: msg.ownershipRefreshIntervalMs,
      });

    if (!shouldReport) {
      const aborted = await search!.run({ visits: maxVisits, maxTimeMs, batchSize, shouldAbort });
      if (aborted || shouldAbort()) {
        postCanceled();
        if (msg.reuseTree !== true) {
          search = null;
          searchKey = null;
        }
        return;
      }
      postAnalysis(buildAnalysis(), 'katago:analyze_result');
      if (msg.reuseTree !== true) {
        search = null;
        searchKey = null;
      }
      return;
    }

    const deadline = getAnimationNow() + maxTimeMs;
    let lastReportVisits = -1;
    while (true) {
      if (shouldAbort()) {
        postCanceled();
        if (msg.reuseTree !== true) {
          search = null;
          searchKey = null;
        }
        return;
      }
      const now = getAnimationNow();
      const remaining = deadline - now;
      if (remaining <= 0) break;
      const sliceMs = Math.min(reportEveryMs, remaining);
      const aborted = await search!.run({ visits: maxVisits, maxTimeMs: sliceMs, batchSize, shouldAbort });
      if (aborted || shouldAbort()) {
        postCanceled();
        if (msg.reuseTree !== true) {
          search = null;
          searchKey = null;
        }
        return;
      }
      const analysis = buildAnalysis();
      const done = analysis.rootVisits >= maxVisits || getAnimationNow() >= deadline;
      if (done) {
        postAnalysis(analysis, 'katago:analyze_result');
        if (msg.reuseTree !== true) {
          search = null;
          searchKey = null;
        }
        return;
      }
      if (analysis.rootVisits > lastReportVisits) {
        lastReportVisits = analysis.rootVisits;
        postAnalysis(analysis, 'katago:analyze_update');
      }
    }

    postAnalysis(buildAnalysis(), 'katago:analyze_result');
    if (msg.reuseTree !== true) {
      search = null;
      searchKey = null;
    }
  }
}

self.onmessage = (ev: MessageEvent<KataGoWorkerRequest>) => {
  const msg = ev.data;
  if (msg.type === 'katago:cancel_analyses') {
    // Handled here, not on the queue — the queue may be blocked by the very
    // search being canceled. Marking every group stale flips the running
    // search's shouldAbort() and cancels anything queued behind it.
    for (const group of latestAnalyzeByGroup.keys()) latestAnalyzeByGroup.set(group, -1);
    interactiveToken++;
    return;
  }
  if (msg.type === 'katago:analyze') {
    const analysisGroup = msg.analysisGroup ?? 'background';
    latestAnalyzeByGroup.set(analysisGroup, msg.id);
    if (analysisGroup === 'interactive') interactiveToken++;
    analyzeMeta.set(msg, { analysisGroup, interactiveToken });
  }
  queue = queue
    .then(() => handleMessage(msg))
    .catch((err: unknown) => {
      if (msg.type === 'katago:init') {
        post({
          type: 'katago:init_result',
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (msg.type === 'katago:eval') {
        post({
          type: 'katago:eval_result',
          id: msg.id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (msg.type === 'katago:human_policy') {
        post({
          type: 'katago:human_policy_result',
          id: msg.id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (msg.type === 'katago:eval_batch') {
        post({
          type: 'katago:eval_batch_result',
          id: msg.id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (msg.type === 'katago:analyze') {
        post({
          type: 'katago:analyze_result',
          id: msg.id,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
    });
};
