import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams, useLocation } from 'react-router-dom';
import { useAuth } from '../auth';
import { Board, type Annotation } from '../Board';
import { playMove, replay } from '../goRules';
import { movesFromSgf, setupStonesFromSgf, sgfInfo, toSgf } from '../sgf';
import { gameDate, gameEvent, gameOutcome, getGame, saveGame } from '../data/games';
import { listFoxAccounts } from '../data/fox';
import { loadReview, newReviewId, saveReview } from '../data/reviews';
import type { GameDoc, GameMove, PlayDefaults, SavedNode } from '../data/model';
import type { Color, Stone } from '../types';
import {
  addMove, buildTree, depthOf, deserializeVariations, extendMainline, leafOf, movesTo,
  nodeAtDepth, pathIds, pruneSubtree, serializeVariations, variationLines, type GameTree,
} from '../variations';
import { genmoveBrowser, rankLabel, scoreTrajectory, type WebAnalysis } from '../katago/webEngine';
import { genmove } from '../data/katago';
import { useAnalysisSession, PONDER_TARGET } from '../katago/useAnalysisSession';
import { useEngineHub } from '../katago/engineHub';
import { useEngineLease } from '../katago/engineLease';
import { Spinner } from '../Spinner';
import { downloadSgf } from '../sgfDownload';
import { ReviewGraph } from '../ReviewGraph';
import './GameReview.css';

const COLS = 'ABCDEFGHJKLMNOPQRST';
const coordLabel = (x: number, y: number) => `${COLS[x]}${19 - y}`;
const scoreLabel = (lead: number) => `${lead >= 0 ? 'B' : 'W'}+${Math.abs(lead).toFixed(1)}`;
const other = (c: Color): Color => (c === 'B' ? 'W' : 'B');

type Point = { move: number; lead: number };
type Mode = 'play' | 'review';

const OFFLINE_MSG = 'Could not run KataGo — your browser may not support WebGPU, or the model failed to load.';

/** Whether a score trips the play-mode alert, from your side's perspective.
 * `base` is the estimate `dropMoves` earlier — only the drop trigger needs it. */
function alertFires(
  play: PlayDefaults, myColor: Color, lead: number | null, base: number | null,
): boolean {
  if (play.scoreMode !== 'alert' || lead === null) return false;
  const mine = myColor === 'B' ? lead : -lead;
  if (play.alertKind === 'drop') {
    if (base === null) return false;
    return (myColor === 'B' ? base : -base) - mine >= (play.dropPoints ?? 5);
  }
  return -mine >= play.alertThreshold;
}

/** Most recent estimate at or before depth `d` on `leaf`'s line, from a
 * node-keyed score map — the walk-back `scoreBefore` does over a drawn curve. */
function leadAtOrBefore(
  tree: GameTree, leaf: number, scores: Record<number, number>, d: number,
): number | null {
  for (let k = Math.max(0, d); k >= 0; k--) {
    const v = scores[nodeAtDepth(tree, leaf, k)];
    if (v !== undefined) return v;
  }
  return null;
}

/** Most recent recorded estimate at or before `move`, else null. */
function scoreBefore(points: Point[], move: number): number | null {
  let best: number | null = null;
  for (const p of points) {
    if (p.move <= move) best = p.lead; else break;
  }
  return best;
}

/** The board surface, in two modes. Review: step the game, branch variations,
 * analyze. Play: the human-like net answers your moves at the leaf of whatever
 * line you're on — so a game you play from a reviewed position is just another
 * variation of it. `shared` is the read-only view behind a public link: no
 * account, no persistence, analysis off until the viewer asks for it. `fresh`
 * starts from an empty board (the /play entry), where
 * the first line you play becomes the game's mainline and Save persists it. */
export function GameReview({ fresh = false, shared = false }: {
  fresh?: boolean;
  shared?: boolean;
}) {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const { user, profile } = useAuth();
  // Where "← Games" returns — the games-list page you came from, else the list.
  const backTo = (location.state as { from?: string } | null)?.from ?? '/review';
  const [loaded, setLoaded] = useState<{ id: string; game: GameDoc | null } | null>(null);
  const [myUids, setMyUids] = useState<Set<number>>(new Set());
  const [cursor, setCursor] = useState(0);
  // Analysis is on whenever you're in review mode; play mode never analyzes.
  const [mode, setMode] = useState<Mode>(fresh ? 'play' : 'review');
  // A shared viewer opts in: the net is a ~98 MB download nobody should pay
  // for just to look at a game.
  const [analyzeOn, setAnalyzeOn] = useState(!shared);
  const { model, visits, batchOverride, engineReady, leaseStatus, play, openSettings } = useEngineHub();
  // Play mode: the side you're taking, the human net's own score estimates
  // (keyed by node, like analyzedScores), and the alert-mode reveal.
  const [myColor, setMyColor] = useState<Color>('B');
  const [playScores, setPlayScores] = useState<Record<number, number>>({});
  const [alerted, setAlerted] = useState(false);
  const [replyAt, setReplyAt] = useState<number | null>(null);
  const [offline, setOffline] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // Pondering raises the session's search target to "effectively forever";
  // the same search just keeps deepening.
  const [ponder, setPonder] = useState(false);
  // Live score estimates, keyed by tree node id (not move depth) so they survive
  // line switches and a variation reuses the mainline's cached prefix.
  const [analyzedScores, setAnalyzedScores] = useState<Record<number, number>>({});
  // Full per-position KataGo output, keyed `node:model`, so scrubbing back to a
  // seen position is instant. Cleared on Rerun and on game change.
  const [analysisCache, setAnalysisCache] = useState<Map<string, WebAnalysis>>(() => new Map());
  // Settings signature the cached trajectory was computed for — null until it
  // runs, and reset whenever the curve is recomputed from scratch.
  const [trajFor, setTrajFor] = useState<string | null>(null);
  const [trajRunning, setTrajRunning] = useState(false);
  const trajRanRef = useRef(false);          // gate (non-reactive, avoids self-abort)
  const [rerunToken, setRerunToken] = useState(0);   // bump to force a recompute
  // Variation tree. `line` is the leaf that defines the line currently on
  // screen; `cursor` is how far along that line we're at. The tree persists to
  // `reviews/{reviewId}` (owner-only) via a debounced, fire-and-forget writer.
  const [tree, setTree] = useState<GameTree | null>(null);
  const [line, setLine] = useState(0);
  const lineRef = useRef<{ moves: GameMove[]; nodeIds: number[] } | null>(null);
  const reviewIdRef = useRef<string | null>(null);
  const reviewCreatedRef = useRef(0);
  const lastSavedRef = useRef('[]');   // JSON of the last-persisted nodes
  const dirtyRef = useRef<{ json: string; nodes: SavedNode[]; gameId: string; ownerUid: string } | null>(null);
  const saveTimerRef = useRef<number | null>(null);

  // Board square is sized from the content area so the whole review — header,
  // graph, board, controls, move list — fits one screen without page scroll.
  // Below a narrow width the columns stack and the body scrolls internally.
  const [boardSize, setBoardSize] = useState(480);
  const [stacked, setStacked] = useState(false);
  const bodyObs = useRef<ResizeObserver | null>(null);
  const bodyRef = useCallback((el: HTMLDivElement | null) => {
    bodyObs.current?.disconnect();
    if (!el) return;
    const measure = () => {
      const cs = getComputedStyle(el);
      const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
      const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
      const gap = parseFloat(cs.columnGap) || 20;
      const availW = el.clientWidth - padX;
      const availH = el.clientHeight - padY;
      const stack = availW < 720;
      const side = stack
        ? Math.max(280, Math.min(availW, availH * 0.82, 560))
        : Math.max(300, Math.min(availH, availW - (availW < 1000 ? 320 : 380) - gap, 820));
      setBoardSize(Math.floor(side));
      setStacked(stack);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    bodyObs.current = ro;
  }, []);

  useEffect(() => {
    if (fresh) return;
    let active = true;
    getGame(id ?? '')
      .then((g) => { if (active) setLoaded({ id: id ?? '', game: g }); })
      .catch(() => { if (active) setLoaded({ id: id ?? '', game: null }); });   // denied / missing → "not found"
    return () => { active = false; };
  }, [id, fresh]);

  // A game being played from an empty board: a real GameDoc shape with no id,
  // so every downstream memo works unchanged. Save turns it into a stored game.
  const [freshGame] = useState<GameDoc | null>(() => (fresh
    ? {
        id: 'new', ownerUid: '', source: 'go-training', createdAt: Date.now(),
        sgf: toSgf([], { komi: 7.5, rules: 'Chinese' }),
      } satisfies GameDoc
    : null));

  const loading = !fresh && (!loaded || loaded.id !== id);
  // A fresh game has no id of its own, so there is nothing to persist against
  // until it is saved.
  const game = freshGame ?? (loading ? null : loaded?.game ?? null);
  const mainlineMoves = useMemo(() => (game ? movesFromSgf(game.sgf) : []), [game]);
  // Setup (handicap) stones seed every board reconstruction and analysis.
  const setup = useMemo(() => (game ? setupStonesFromSgf(game.sgf) : []), [game]);

  // Which participant (if any) is one of the game owner's own accounts — for the
  // win/loss accent. Readable by the owner and, per the rules, a linked teacher.
  useEffect(() => {
    if (!game || game.source !== 'fox' || !user) return;
    let on = true;
    listFoxAccounts(game.ownerUid)
      .then((a) => { if (on) setMyUids(new Set(a.filter((x) => x.isMine).map((x) => x.uid))); })
      .catch(() => { if (on) setMyUids(new Set()); });
    return () => { on = false; };
  }, [game, user]);

  // (Re)build the variation tree whenever a different game is shown; start the
  // cursor at the end of the mainline (render-time adjustment, not an effect).
  const [treeForGame, setTreeForGame] = useState<GameDoc | null>(null);
  if (game && game !== treeForGame) {
    setTreeForGame(game);
    const t = buildTree(mainlineMoves);
    setTree(t);
    setLine(t.mainlineLeafId);
    setCursor(mainlineMoves.length);
    setAnalyzedScores({});
    setAnalysisCache(new Map());
    setTrajFor(null);
  }

  // Reset per-game refs for a newly-shown game (refs can't be set during render).
  useEffect(() => { trajRanRef.current = false; }, [game]);

  // Write any pending variation edit now (debounce fire, unmount, game switch).
  const flush = useCallback(() => {
    if (saveTimerRef.current != null) { clearTimeout(saveTimerRef.current); saveTimerRef.current = null; }
    const d = dirtyRef.current;
    if (!d) return;
    dirtyRef.current = null;
    if (!reviewIdRef.current) { reviewIdRef.current = newReviewId(); reviewCreatedRef.current = Date.now(); }
    lastSavedRef.current = d.json;
    saveReview({
      id: reviewIdRef.current, ownerUid: d.ownerUid, gameId: d.gameId,
      nodes: d.nodes, createdAt: reviewCreatedRef.current, updatedAt: Date.now(),
    }).catch(() => { lastSavedRef.current = 'retry'; /* never equals real JSON — retries on the next edit */ });
  }, []);

  // Load the owner's saved variations for this game, splicing them into the
  // mainline tree. Flush any pending write before switching games. Skipped
  // while the game is unsaved — it has no id to hang a review on.
  useEffect(() => {
    if (fresh || shared || !user || !game) return;
    let active = true;
    const ownerUid = user.uid;
    const gameId = game.id;
    reviewIdRef.current = null;
    reviewCreatedRef.current = 0;
    lastSavedRef.current = '[]';
    loadReview(ownerUid, gameId)
      .then((review) => {
        if (!active || !review) return;
        const restored = deserializeVariations(mainlineMoves, review.nodes);
        reviewIdRef.current = review.id;
        reviewCreatedRef.current = review.createdAt;
        lastSavedRef.current = JSON.stringify(serializeVariations(restored));
        setTree(restored);
      })
      .catch(() => { /* keep the session-only tree on failure */ });
    return () => { active = false; flush(); };
  }, [fresh, shared, user, game, mainlineMoves, flush]);

  // Debounce a persist whenever the tree gains/loses variation nodes.
  useEffect(() => {
    if (fresh || shared || !user || !game || !tree) return;
    const nodes = serializeVariations(tree);
    const json = JSON.stringify(nodes);
    if (json === lastSavedRef.current) { dirtyRef.current = null; return; }
    dirtyRef.current = { json, nodes, gameId: game.id, ownerUid: user.uid };
    if (saveTimerRef.current != null) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(flush, 1000);
  }, [tree, fresh, shared, user, game, flush]);

  // Flush a pending write when leaving the page.
  useEffect(() => flush, [flush]);

  // The moves + node ids along the line currently on screen, and the mainline's.
  const lineMoves = useMemo(() => (tree ? movesTo(tree, line) : []), [tree, line]);
  const lineNodeIds = useMemo(() => (tree ? pathIds(tree, line) : []), [tree, line]);
  const mainNodeIds = useMemo(() => (tree ? pathIds(tree, tree.mainlineLeafId) : []), [tree]);
  const lines = useMemo(() => (tree ? variationLines(tree) : []), [tree]);

  // What the score trajectory analyzes — kept in a ref so it isn't a dependency.
  useEffect(() => { lineRef.current = { moves: lineMoves, nodeIds: lineNodeIds }; }, [lineMoves, lineNodeIds]);
  const total = lineMoves.length;
  const onMainline = !!tree && line === tree.mainlineLeafId;
  // Where the current line leaves the mainline (move number after which it
  // diverges); -1 on the mainline itself.
  const branchPoint = useMemo(() => {
    if (!tree || onMainline) return -1;
    const off = lineNodeIds.find((nid) => !tree.nodes[nid].mainline);
    return off != null ? depthOf(tree, off) - 1 : -1;
  }, [tree, lineNodeIds, onMainline]);

  // Score curve for the current line: each depth's node from the (node-keyed)
  // caches, so the shared prefix of a variation comes straight from the analysis
  // already done on the line it branched from.
  const points = useMemo<Point[]>(() => {
    if (!tree) return [];
    const out: Point[] = [];
    for (let i = 0; i <= total; i++) {
      const node = lineNodeIds[i];
      // Prefer the mode's own estimator, then fall back: a line played out keeps
      // the analyzed prefix behind it, and a reviewed line keeps the estimates
      // recorded while it was played.
      let lead: number | undefined = mode === 'play'
        ? playScores[node] ?? analyzedScores[node]
        : analyzedScores[node] ?? playScores[node];
      if (lead === undefined && tree.nodes[node]?.mainline) lead = game?.scoreAt?.[String(i)];
      if (lead !== undefined) out.push({ move: i, lead });
    }
    return out;
  }, [tree, lineNodeIds, total, mode, playScores, analyzedScores, game]);

  const shown = useMemo(() => replay(lineMoves.slice(0, cursor), setup), [lineMoves, cursor, setup]);

  // Keep the active move visible by scrolling only the move-list container —
  // never the page (scrollIntoView would drag the whole layout up when the list
  // is near the bottom). Accounts for the sticky header height.
  const activeRef = useRef<HTMLTableRowElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const row = activeRef.current;
    const box = listRef.current;
    if (!row || !box) return;
    const rowRect = row.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();
    const headH = box.querySelector('thead')?.getBoundingClientRect().height ?? 0;
    if (rowRect.top < boxRect.top + headH) box.scrollTop -= boxRect.top + headH - rowRect.top;
    else if (rowRect.bottom > boxRect.bottom) box.scrollTop += rowRect.bottom - boxRect.bottom;
  }, [cursor, line]);

  // Step navigation (arrows and the ⏮◀▶⏭ buttons). Stepping back to or past
  // the branch point exits the variation — the position is in the shared
  // prefix, so stepping forward again follows the game line, as if a Game-
  // column move had been clicked.
  const stepTo = useCallback((m: number) => {
    setReplyAt(null);   // navigating is not playing — the opponent stays put
    const c = Math.max(0, Math.min(total, m));
    if (tree && line !== tree.mainlineLeafId && c <= branchPoint) setLine(tree.mainlineLeafId);
    setCursor(c);
  }, [total, tree, line, branchPoint]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') stepTo(cursor - 1);
      else if (e.key === 'ArrowRight') stepTo(cursor + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cursor, stepTo]);

  // Whose turn it is at the cursor: the color of the line's next move if there
  // is one, else the opposite of the last played move (Black on an empty board).
  const toPlay: Color =
    cursor < lineMoves.length ? lineMoves[cursor].color
      : cursor > 0 ? other(lineMoves[cursor - 1].color)
        : 'B';

  // Play (or re-walk) a move at the cursor — branching a variation when it
  // departs from the line, or advancing when it matches an existing child.
  // Shared by the board and by the opponent's replies.
  const commitMove = useCallback((x: number, y: number, color: Color): number | null => {
    if (!tree) return null;
    const legal = playMove(shown.stones, color, x, y, shown.koPoint);
    if (!legal.ok) return null;
    const branchNode = nodeAtDepth(tree, line, cursor);
    // An unsaved game has no source SGF, so the first line played is the game
    // itself: extend the mainline. Everywhere else a move branches.
    const { tree: next, childId } = fresh && branchNode === tree.mainlineLeafId
      ? extendMainline(tree, { color, x, y })
      : addMove(tree, branchNode, { color, x, y });
    const stayOnLine = lineNodeIds.includes(childId);
    // Branching = departing from a line that already continued past here.
    // Extending the leaf of the line you're on is not branching.
    const branched = !stayOnLine && tree.nodes[branchNode].children.length > 0;
    setTree(next);
    setLine(stayOnLine ? line : leafOf(next, childId));
    setCursor(depthOf(next, childId));
    // Committing to a new line hides the alert graph until you fall behind again.
    if (branched) setAlerted(false);
    return childId;
  }, [tree, shown, line, cursor, lineNodeIds, fresh]);

  // ---- play mode ----
  // The opponent answers a move you *played*, never a position you merely
  // navigated to: `replyAt` is the node it owes a reply at, set when you play
  // (or hand it your colour) and cleared by any navigation. Re-walking a move
  // the game already contains is still playing it, so it counts.
  const currentNode = tree ? nodeAtDepth(tree, line, cursor) : -1;
  const atLeaf = cursor === total;
  const playEngine = play.engine ?? 'browser';
  const aiTurn = mode === 'play' && !offline && toPlay !== myColor && currentNode === replyAt;
  const playLease = useEngineLease(mode === 'play' && playEngine === 'browser');

  useEffect(() => {
    if (!aiTurn || !tree) return;
    if (playEngine === 'browser' && playLease !== 'active') return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ctrl = new AbortController();
    const startedAt = Date.now();
    const atNode = currentNode;
    const movesSoFar = lineMoves.slice(0, cursor);
    const gen = playEngine === 'local'
      ? genmove({
          initialStones: setup, moves: movesSoFar, initialPlayer: movesSoFar[0]?.color ?? 'B',
          rank: play.rank, temperature: play.temperature, signal: ctrl.signal,
        }).then((r) => ({ move: r.move, scoreLead: r.root.score_lead }))
      : genmoveBrowser({
          stones: shown.stones,
          previousStones: cursor > 0 ? replay(lineMoves.slice(0, cursor - 1), setup).stones : undefined,
          moves: movesSoFar,
          toPlay,
          rank: play.rank,
          temperature: play.temperature,
          komi: 7.5,
          koPoint: shown.koPoint,
        });
    gen
      .then((res) => {
        if (!active) return;
        // Minimum "think time" so the reply doesn't snap in instantly.
        const wait = Math.max(0, play.moveDelay * 1000 - (Date.now() - startedAt));
        timer = setTimeout(() => {
          if (!active) return;
          setPlayScores((s) => ({ ...s, [atNode]: res.scoreLead }));
          setOffline(false);
          // Falling behind reveals the graph, and it stays up while you look
          // around — until you commit to a new line (see commitMove).
          const base = leadAtOrBefore(tree, line, playScores, cursor - (play.dropMoves ?? 10));
          if (alertFires(play, myColor, res.scoreLead, base)) setAlerted(true);
          if (res.move) commitMove(res.move.x, res.move.y, toPlay);
        }, wait);
      })
      .catch(() => { if (active && !ctrl.signal.aborted) setOffline(true); });
    return () => { active = false; ctrl.abort(); if (timer) clearTimeout(timer); };
  }, [aiTurn, tree, line, cursor, currentNode, lineMoves, shown, setup, toPlay, playEngine,
    playLease, play, myColor, playScores, retryToken, commitMove]);

  // Score from your perspective at the cursor (positive = you're ahead), and the
  // numbers the alert banner quotes: an absolute deficit, or points lost recently.
  const cursorLead = scoreBefore(points, cursor);
  const userLead = cursorLead === null ? null : (myColor === 'B' ? cursorLead : -cursorLead);
  const dropMoves = play.dropMoves ?? 10;
  const baseLead = scoreBefore(points, Math.max(0, cursor - dropMoves));
  const baseUserLead = baseLead === null ? null : (myColor === 'B' ? baseLead : -baseLead);
  const behindBy = userLead === null ? null : -userLead;
  const droppedBy = userLead === null || baseUserLead === null ? null : baseUserLead - userLead;
  const alerting = mode === 'play' && alertFires(play, myColor, cursorLead, baseLead);

  // Session analysis of the current position (opt-in): one streaming search
  // per position; snapshots arrive as it deepens, and the ponder toggle raises
  // the target so the same search keeps going.
  const nextMove = cursor < lineMoves.length ? lineMoves[cursor] : null;
  const sessionPosition = useMemo(() => {
    if (!game || !tree) return null;
    const forNode = nodeAtDepth(tree, line, cursor);
    const childStones = nextMove ? replay(lineMoves.slice(0, cursor + 1), setup).stones : null;
    return {
      positionId: `${id ?? 'new'}:${line}:${cursor}:${model.id}`,
      parentPositionId: cursor > 0 ? `${id ?? 'new'}:${line}:${cursor - 1}:${model.id}` : undefined,
      stones: shown.stones,
      previousStones: cursor > 0 ? replay(lineMoves.slice(0, cursor - 1), setup).stones : undefined,
      previousPreviousStones: cursor > 1 ? replay(lineMoves.slice(0, cursor - 2), setup).stones : undefined,
      initialStones: setup,
      moves: lineMoves.slice(0, cursor),
      toPlay,
      evalNext: nextMove && childStones ? { move: { x: nextMove.x, y: nextMove.y }, stones: childStones } : null,
      nodeId: forNode,
      nodeKey: `${forNode}:${model.id}`,
    };
  }, [game, tree, id, line, cursor, model.id, shown, setup, toPlay, lineMoves, nextMove]);

  const analyzing = mode === 'review' && analyzeOn;
  const session = useAnalysisSession({
    enabled: analyzing && engineReady && !!sessionPosition,
    model,
    position: sessionPosition,
    targetVisits: ponder ? PONDER_TARGET : visits,
    batchSize: batchOverride ?? undefined,
    onSnapshot: (a, pos) => {
      if (pos.nodeKey) {
        setAnalysisCache((m) => {
          const prev = m.get(pos.nodeKey!);
          if (prev && prev.rootVisits > a.rootVisits) return m;
          return new Map(m).set(pos.nodeKey!, a);
        });
      }
      const node = pos.nodeId;
      if (node !== undefined) {
        setAnalyzedScores((s) => (s[node] === a.rootScoreLead ? s : { ...s, [node]: a.rootScoreLead }));
      }
    },
  });

  // WebGPU can't allocate this model/position on some GPUs — point at the
  // lighter model / batch setting rather than surfacing the raw GPU error.
  const analysisErr = session.error
    ? (/createBuffer|GPUDevice|too large|out of memory/i.test(session.error)
      ? 'this GPU couldn’t run the model here — use a smaller model or reduce batch size (AI engine button in the sidebar)'
      : session.error)
    : '';

  // Score curve over the whole line on screen, filling the node-keyed cache.
  // Runs when review mode is entered and on Rerun; branching a variation can't
  // abort a running pass, because the line it walks is read from a ref rather
  // than tracked as a dependency.
  const trajSig = `${model.id}:${visits}`;
  useEffect(() => {
    if (!analyzing || !game || !engineReady) return;
    if (trajRanRef.current) return;   // already covered this line; re-armed by Rerun
    trajRanRef.current = true;
    setTrajFor(trajSig);
    setTrajRunning(true);
    let active = true;
    const ctrl = new AbortController();
    // The line on screen — the game's mainline until you branch off it, and the
    // line you played when that's what you're looking at. Read through the ref
    // so the tree isn't a dependency: branching must not abort a running pass.
    const passMoves = lineRef.current?.moves ?? mainlineMoves;
    const passNodes = lineRef.current?.nodeIds ?? mainlineMoves.map((_, k) => k + 1);
    const passTotal = passMoves.length;
    const boards: Stone[][] = [setup.map((st) => ({ x: st.x, y: st.y, color: st.color }))];
    let stones: Stone[] = boards[0];
    let ko: { x: number; y: number } | null = null;
    for (let k = 0; k < passTotal; k++) {
      const mv = passMoves[k];
      if (mv.x < 0 || mv.y < 0) { boards.push(stones); continue; } // pass
      const r = playMove(stones, mv.color, mv.x, mv.y, ko);
      if (!r.ok) { boards.push(stones); continue; }
      stones = r.stones; ko = r.koPoint;
      boards.push(stones);
    }
    const positions = boards.map((b, k) => ({
      stones: b,
      previousStones: k > 0 ? boards[k - 1] : undefined,
      previousPreviousStones: k > 1 ? boards[k - 2] : undefined,
      moves: passMoves.slice(0, k),
      toPlay: (k < passTotal ? passMoves[k].color : k > 0 ? other(passMoves[k - 1].color) : 'B') as Color,
    }));
    scoreTrajectory({
      model,
      positions,
      initialStones: setup,
      komi: 7.5,
      // Positions per forward pass; omit for Auto (latency-budgeted from the
      // measured forward-pass time). Smaller batches also shrink the peak WebGPU
      // buffer for GPUs that refuse large mappedAtCreation allocations.
      chunk: batchOverride ?? undefined,
      onChunk: (from, scores) => {
        setAnalyzedScores((s) => {
          const next = { ...s };
          scores.forEach((v, j) => {
            const node = passNodes[from + j];
            if (node !== undefined) next[node] = v;
          });
          return next;
        });
      },
      signal: ctrl.signal,
    })
      .catch(() => { trajRanRef.current = false; /* aborted / engine error — allow a later run */ })
      .finally(() => { if (active) setTrajRunning(false); });
    return () => { active = false; ctrl.abort(); };
  }, [analyzing, engineReady, game, model, visits, mainlineMoves, setup, batchOverride, rerunToken, trajSig]);

  if (loading) return <div className="center-screen"><Spinner /></div>;
  if (!game || !tree) {
    return (
      <div className="gr">
        <p>Game not found.</p>
        <Link to={backTo}>← Back to games</Link>
      </div>
    );
  }

  const seek = (m: number) => { setReplyAt(null); setCursor(Math.max(0, Math.min(total, m))); };

  // Switching modes never moves the board — you keep the position you're on.
  // Review always arrives with the AI on; play never analyzes.
  const enterMode = (m: Mode) => {
    setMode(m);
    setOffline(false);
    setReplyAt(null);
    if (m === 'review') {
      setAnalyzeOn(true);
      rerun();   // the line has moved on since the last pass — recompute it
    }
    else setMyColor(toPlay);   // you take the side to move where you entered
  };

  // In play mode the board is yours alone; in review either colour can be played.
  const boardPlay = (x: number, y: number) => {
    if (mode === 'play' && toPlay !== myColor) return;
    const child = commitMove(x, y, toPlay);
    if (mode === 'play' && child !== null) setReplyAt(child);
  };

  // Persist a game played from an empty board: the mainline becomes the game,
  // any branches off it become its variations, and the view moves to the saved
  // game so everything from there on auto-persists.
  const saveFresh = async () => {
    if (!user || !fresh || saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      const mainNodes = pathIds(tree, tree.mainlineLeafId);
      const moves = movesTo(tree, tree.mainlineLeafId);
      const scoreAt: Record<string, number> = {};
      mainNodes.forEach((nid, depth) => {
        if (playScores[nid] !== undefined) scoreAt[String(depth)] = playScores[nid];
      });
      const rankShort = play.rank.replace('rank_', '');
      const black = myColor === 'B';
      const myName = profile?.displayName || 'Me';
      const oppName = 'Human-like KataGo';
      const createdAt = Date.now();
      const saved = await saveGame({
        ownerUid: user.uid,
        source: 'go-training',
        createdAt,
        myColor,
        rank: play.rank,
        rankLabel: rankLabel(play.rank),
        temperature: play.temperature,
        sgf: toSgf(moves, {
          komi: 7.5,
          rules: 'Chinese',
          playerBlack: black ? myName : oppName,
          playerWhite: black ? oppName : myName,
          rankBlack: black ? '?' : rankShort,
          rankWhite: black ? rankShort : '?',
          date: new Date(createdAt).toISOString().slice(0, 10),
        }),
        scoreAt,
        moveCount: moves.length,
        finalScore: leadAtOrBefore(tree, tree.mainlineLeafId, playScores, moves.length),
      });
      const nodes = serializeVariations(tree);
      if (nodes.length) {
        await saveReview({
          id: newReviewId(), ownerUid: user.uid, gameId: saved.id,
          nodes, createdAt, updatedAt: createdAt,
        });
      }
      navigate(`/review/${saved.id}`, { replace: true });
    } catch {
      setSaveError('Could not save the game.');
      setSaving(false);
    }
  };

  // Delete a variation subtree (a chip, or a variation move onward). If the
  // current line ran through it, land on the surviving parent's line — the
  // variation's remaining prefix when truncating, else the mainline.
  const deleteBranch = (nodeId: number) => {
    const parent = tree.nodes[nodeId]?.parent ?? null;
    const next = pruneSubtree(tree, nodeId);
    setTree(next);
    if (!next.nodes[line]) {
      const fallback = parent != null && next.nodes[parent] ? leafOf(next, parent) : next.mainlineLeafId;
      setLine(fallback);
      setCursor((c) => Math.min(c, depthOf(next, fallback)));
    }
  };
  // Recompute the score graph for the current line from scratch (on entering
  // review, or after changing model/visits).
  const rerun = () => {
    trajRanRef.current = false;
    setAnalysisCache(new Map());
    setTrajFor(null);
    setAnalyzedScores({});
    setRerunToken((t) => t + 1);
  };
  const trajStale = trajFor !== null && trajFor !== trajSig;
  const clearLines = () => {
    const t = buildTree(mainlineMoves);
    setTree(t);
    setLine(t.mainlineLeafId);
    setCursor((c) => Math.min(c, mainlineMoves.length));
  };

  // Move-table navigation. Any Game-column click returns to the mainline at that
  // move (collapsing the variation); clicking a variation move seeks within it;
  // a preview chip drills into that line. Step navigation (arrows/buttons)
  // exits a variation once it crosses back over the branch point; the graph
  // and scrub bar seek within the current line.
  const goGame = (depth: number) => { setReplyAt(null); setLine(tree.mainlineLeafId); setCursor(depth); };
  const goVar = (depth: number) => { setReplyAt(null); setCursor(depth); };
  const enterAt = (leafId: number, depth: number) => { setReplyAt(null); setLine(leafId); setCursor(depth); };

  // What the score is allowed to reveal. Play mode honours the score setting:
  // 'hide' shows nothing, 'alert' only once you've been warned, 'show' always.
  const showScore = mode === 'review' ? true : play.scoreMode === 'show';
  const showGraph = mode === 'review'
    ? analyzeOn
    : play.scoreMode === 'show' || (play.scoreMode === 'alert' && (alerting || alerted));

  const mark = cursor > 0 ? lineMoves[cursor - 1] : null;
  const annotations: Annotation[] = mark ? [{ kind: 'circle', x: mark.x, y: mark.y }] : [];
  const cursorScore = scoreBefore(points, cursor);
  const info = sgfInfo(game.sgf);
  const outcome = gameOutcome(game, myUids);
  const mainlineTotal = depthOf(tree, tree.mainlineLeafId);
  // Table shape: rows are move numbers; left = mainline, right = the current
  // variation (or, on the mainline, previews of where variations branch off).
  const mainLen = mainNodeIds.length - 1;
  const maxRows = onMainline ? mainLen : Math.max(mainLen, lineMoves.length);
  const activeIsMain = cursor === 0 || !!tree.nodes[lineNodeIds[cursor]]?.mainline;
  // Non-continuation children at the node before row `i` — variations off the
  // mainline (Game view) or sub-variations off the current line (variation view).
  const previewsAt = (i: number): number[] => {
    if (onMainline) {
      const parent = mainNodeIds[i - 1];
      return parent != null ? tree.nodes[parent].children.filter((c) => !tree.nodes[c].mainline) : [];
    }
    if (i - 1 < branchPoint) return [];   // mainline prefix rows carry no chips
    const parent = lineNodeIds[i - 1];
    const cont = lineNodeIds[i];
    // At the branch row this lists the sibling variations (the mainline
    // continuation already sits in the Game column), so every variation stays
    // reachable — and deletable — while exploring one of them.
    return parent != null
      ? tree.nodes[parent].children.filter((c) => c !== cont && !tree.nodes[c].mainline)
      : [];
  };
  const moveCell = (node: number, depth: number, active: boolean, showNum: boolean, onClick: () => void) => {
    const m = tree.nodes[node].move;
    if (!m) return null;
    const score = analyzedScores[node]
      ?? (tree.nodes[node].mainline ? game.scoreAt?.[String(depth)] : undefined);
    return (
      <button type="button" className={active ? 'gr-mv active' : 'gr-mv'} onClick={onClick}>
        {showNum && <span className="mv-num">{depth}</span>}
        <span className={`mv-color mv-${m.color}`} aria-hidden />
        <span className="mv-coord">{coordLabel(m.x, m.y)}</span>
        {score !== undefined && <span className="mv-score">{scoreLabel(score)}</span>}
      </button>
    );
  };
  const previewChip = (node: number) => {
    const m = tree.nodes[node].move;
    if (!m) return null;
    return (
      <span key={node} className="gr-mv-preview">
        <button
          type="button"
          className="gr-mv-preview-go"
          onClick={() => enterAt(leafOf(tree, node), depthOf(tree, node))}
          title={`Explore variation ${coordLabel(m.x, m.y)}`}
        >
          <span className={`mv-color mv-${m.color}`} aria-hidden />
          <span className="mv-coord">{coordLabel(m.x, m.y)}</span>
        </button>
        <button
          type="button"
          className="gr-mv-preview-del"
          onClick={() => deleteBranch(node)}
          aria-label={`Delete variation ${coordLabel(m.x, m.y)}`}
          title="Delete this variation"
        >
          ×
        </button>
      </span>
    );
  };
  const when = new Date(gameDate(game)).toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  });
  const event = gameEvent(game);
  const cachedAnalysis = analyzing && sessionPosition
    ? analysisCache.get(sessionPosition.nodeKey) ?? null : null;
  const currentAnalysis = analyzing ? (session.snapshot ?? cachedAnalysis) : null;
  const running = analyzing && !currentAnalysis && !analysisErr;
  const playedNext = cursor < total ? lineMoves[cursor] : null;
  const aiCandidates = currentAnalysis
    ? [
        ...currentAnalysis.moves.map((m) => ({ x: m.x, y: m.y, loss: m.pointsLost })),
        // The played move gets its own dot when the search didn't already list it.
        ...(playedNext && currentAnalysis.playedEval
          && !currentAnalysis.moves.some((m) => m.x === playedNext.x && m.y === playedNext.y)
          ? [{ x: playedNext.x, y: playedNext.y, loss: currentAnalysis.playedEval.pointsLost }]
          : []),
      ]
    : undefined;

  // The live-analysis line under the graph — null when there's nothing to show
  // (e.g. at the last move), so the card doesn't reserve an empty row.
  const analysisLine = leaseStatus === 'waiting'
    ? <span className="gr-analyze-wait">KataGo AI is running in another tab or window — turn it off there (or close it) to use it here.</span>
    : running ? <Spinner label="Analyzing…" />
      : analysisErr ? <span className="gr-analyze-err">{analysisErr}</span>
        : (currentAnalysis && playedNext && currentAnalysis.playedEval) ? (() => {
            // pointsLost can be slightly negative (played move beat the search's
            // best at low visits) — sign it, don't prefix "−".
            const loss = currentAnalysis.playedEval.pointsLost;
            return (
              <>played {coordLabel(playedNext.x, playedNext.y)}{' '}
                <span className={loss > 0.05 ? 'gr-loss' : undefined}>
                  ({loss < 0 ? '+' : '−'}{Math.abs(loss).toFixed(1)})
                </span>
              </>
            );
          })()
          : null;

  return (
    <div className={`gr${outcome ? ` gr--${outcome}` : ''}`}>
      <div className="gr-head">
        {shared
          ? <Link to="/" className="gr-back">Go training</Link>
          : <Link to={backTo} className="gr-back">← Games</Link>}
        <h1 className="gr-title">
          {fresh ? <>New game <span className="gr-vs">vs.</span> KataGo <span className="gr-rank">[{rankLabel(play.rank)}]</span></> : (
            <>
              {game.name && <span className="gr-game-name">{game.name}<span className="gr-vs"> — </span></span>}
              {info.playerBlack || 'Black'}{info.rankBlack && <span className="gr-rank"> [{info.rankBlack}]</span>}
              <span className="gr-vs"> vs. </span>
              {info.playerWhite || 'White'}{info.rankWhite && <span className="gr-rank"> [{info.rankWhite}]</span>}
            </>
          )}
        </h1>
        <span className="gr-meta">
          {!fresh && <><span>{when}</span><span className="gr-dot">·</span></>}
          {!fresh && event && event !== game.name && (
            <><span>{event}</span><span className="gr-dot">·</span></>
          )}
          <span>{mainlineTotal} moves</span>
          {game.finalScore != null ? (
            <><span className="gr-dot">·</span><strong>{scoreLabel(game.finalScore)}</strong></>
          ) : info.result ? (
            <><span className="gr-dot">·</span>
              <strong className={outcome ? `gr-result gr-result--${outcome}` : undefined}>{info.result}</strong></>
          ) : null}
          {outcome && (
            <span className={`gr-outcome gr-outcome--${outcome}`}>{outcome === 'win' ? 'You won' : 'You lost'}</span>
          )}
        </span>
        <div className="gr-head-spacer" />
        {!shared && (
          <div className="gr-mode" role="group" aria-label="Mode">
            {(['play', 'review'] as Mode[]).map((m) => (
              <button key={m} type="button" className={mode === m ? 'active' : ''} onClick={() => enterMode(m)}>
                {m === 'play' ? 'Play' : 'Review'}
              </button>
            ))}
          </div>
        )}
        {!shared && mode === 'play' && (
          <>
            <button
              type="button"
              className="gr-side"
              onClick={() => { setMyColor(other(myColor)); setReplyAt(currentNode); }}
              title="Play the other side from here — the opponent takes over the colour you leave"
            >
              You: <span className={`mv-color mv-${myColor}`} aria-hidden /> {myColor === 'B' ? 'Black' : 'White'}
            </button>
            <button type="button" className="gr-gear" onClick={openSettings} title="Opponent settings">⚙</button>
          </>
        )}
        {fresh && mainlineTotal > 0 && (
          <button type="button" className="gr-analyze-btn" onClick={saveFresh} disabled={saving || !user}>
            {saving ? 'Saving…' : 'Save game'}
          </button>
        )}
        {saveError && <span className="gr-analyze-err">{saveError}</span>}
        {mode === 'review' && (
          <button
            type="button"
            className={analyzeOn ? 'gr-analyze-btn active' : 'gr-analyze-btn'}
            onClick={() => setAnalyzeOn((o) => !o)}
          >
            {analyzeOn ? 'AI review: on' : 'AI review'}
          </button>
        )}
        {!fresh && (
          <button
            type="button"
            className="gr-gear"
            onClick={() => downloadSgf(game)}
            title="Download this game as an SGF file"
            aria-label="Download SGF"
          >
            ⤓
          </button>
        )}
        {analyzing && (
          <button
            type="button"
            className={ponder ? 'gr-gear active' : 'gr-gear'}
            onClick={() => setPonder((p) => !p)}
            title={ponder ? 'Pause — stop deepening this position' : 'Keep analyzing — deepen this position for more accuracy'}
            aria-pressed={ponder}
          >
            {ponder ? '⏸' : '▶'}
          </button>
        )}
        {analyzing && currentAnalysis && (
          <span className="gr-visits" title="Playouts behind the current analysis">
            {currentAnalysis.rootVisits.toLocaleString()}
          </span>
        )}
      </div>

      <div className={`gr-body${stacked ? ' gr-body--stacked' : ''}`} ref={bodyRef}>
        <div className="gr-board-square" style={{ width: boardSize, height: boardSize }}>
          <Board
            stones={shown.stones}
            annotations={annotations}
            aiCandidates={aiCandidates}
            spinnerAt={null}
            ghostStone={mode === 'review' && playedNext
              ? { x: playedNext.x, y: playedNext.y, color: playedNext.color } : null}
            onPlay={boardPlay}
          />
        </div>

        <div className="gr-panel">
          {mode === 'play' && (
            <div className="gr-play-bar">
              {alerting && (
                <div className="gr-play-alert">
                  {play.alertKind === 'drop'
                    ? <>You've lost {droppedBy!.toFixed(1)} points over the last {dropMoves} moves — step back and try another line.</>
                    : <>You're behind by {behindBy!.toFixed(1)} points — step back and try another line.</>}
                </div>
              )}
              {offline ? (
                <span className="gr-play-status gr-analyze-err">
                  {playEngine === 'local' ? 'KataGo backend offline — is `make api` running?' : OFFLINE_MSG}
                  <button type="button" className="gr-rerun" onClick={() => { setOffline(false); setRetryToken((n) => n + 1); }}>↻</button>
                </span>
              ) : playLease === 'waiting' ? (
                <span className="gr-play-status">KataGo AI is running in another tab or window — turn it off there to play here.</span>
              ) : (
                <span className="gr-play-status">
                  {aiTurn ? 'KataGo is thinking…'
                    : !atLeaf ? 'Earlier position — play a move to continue from here'
                      : `Your move (${myColor === 'B' ? 'Black' : 'White'})`}
                  {' · '}KataGo {rankLabel(play.rank)}
                </span>
              )}
            </div>
          )}
          {showGraph ? (
            <div className="gr-graph-card">
              {mode === 'review' && <div className="gr-graph-head">
                {trajRunning ? (
                  <span className="gr-rerun gr-rerun-busy"><Spinner label="Analyzing…" /></span>
                ) : (
                  <button
                    type="button"
                    className={`gr-rerun${trajStale ? ' stale' : ''}`}
                    onClick={rerun}
                    aria-label="Recompute the score graph from scratch"
                    title={trajStale ? 'Settings changed — recompute the score graph' : 'Recompute the score graph from scratch'}
                  >
                    ↻
                  </button>
                )}
                <span className="gr-graph-kata">
                  {currentAnalysis && (
                    <span>KataGo <strong>{scoreLabel(currentAnalysis.rootScoreLead)}</strong> · {currentAnalysis.rootVisits}v</span>
                  )}
                </span>
              </div>}
              {points.length > 1 ? (
                <ReviewGraph points={points} total={total} cursor={cursor} onSeek={seek} />
              ) : (
                <div className="gr-graph-empty">
                  {running ? <Spinner label="Analyzing…" />
                    : mode === 'play' ? 'The score line appears as the opponent replies.'
                      : 'The score timeline appears as KataGo analyzes the game.'}
                </div>
              )}
              {mode === 'review' && analysisLine && <div className="gr-analysis">{analysisLine}</div>}
            </div>
          ) : (
            <div className="gr-scrub-bar">
              <input type="range" min={0} max={total} value={cursor} onChange={(e) => seek(Number(e.target.value))} aria-label="Move" />
            </div>
          )}

          <div className="gr-controls">
            <button type="button" onClick={() => stepTo(0)} disabled={cursor === 0} aria-label="Start">⏮</button>
            <button type="button" onClick={() => stepTo(cursor - 1)} disabled={cursor === 0} aria-label="Previous">◀</button>
            <button type="button" onClick={() => stepTo(cursor + 1)} disabled={cursor === total} aria-label="Next">▶</button>
            <button type="button" onClick={() => stepTo(total)} disabled={cursor === total} aria-label="End">⏭</button>
            <div className="gr-controls-spacer" />
            <div className="gr-readout">
              <div className="gr-readout-move">
                move {cursor} / {total}{!onMainline && <> · <span className="gr-status-var">variation</span></>}
              </div>
              {cursorScore !== null && showScore && (
                <div className="gr-readout-est">estimate <strong>{scoreLabel(cursorScore)}</strong></div>
              )}
            </div>
          </div>

          <div className="gr-moves-panel" ref={listRef}>
            <table className="gr-moves">
              <thead>
                <tr>
                  <th>Game</th>
                  <th>{onMainline ? 'Variations' : `Variation · from move ${branchPoint > 0 ? branchPoint : 'start'}`}</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: maxRows }, (_, k) => k + 1).map((i) => {
                  const gameNode = i <= mainLen ? mainNodeIds[i] : null;
                  const varNode = !onMainline && i > branchPoint && i < lineNodeIds.length ? lineNodeIds[i] : null;
                  const previews = previewsAt(i);
                  const rowActive = cursor === i;
                  return (
                    <tr key={i} ref={rowActive ? activeRef : null}>
                      <td className="gr-cell">
                        {gameNode != null && moveCell(gameNode, i, rowActive && activeIsMain, true, () => goGame(i))}
                      </td>
                      <td className="gr-cell gr-cell-var">
                        {varNode != null && (
                          <span className="gr-mv-var">
                            {moveCell(varNode, i, rowActive && !activeIsMain, gameNode == null, () => goVar(i))}
                            <button
                              type="button"
                              className="gr-mv-preview-del"
                              onClick={() => deleteBranch(varNode)}
                              aria-label="Delete the variation from this move"
                              title="Delete the variation from this move"
                            >
                              ×
                            </button>
                          </span>
                        )}
                        {previews.length > 0 && (
                          <span className="gr-previews">{previews.map((c) => previewChip(c))}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              {lines.length > 0 && (
                <tfoot>
                  <tr>
                    <td colSpan={2} className="gr-moves-foot">
                      <button type="button" onClick={clearLines}>Clear all variations</button>
                    </td>
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
