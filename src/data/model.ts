// Shared types for the static library and the Firestore documents. The
// Firestore shapes mirror docs/migration-spec.md.

import type { Color } from '../types';

export type Verdict = 'correct' | 'incorrect' | 'flag';

// ---------- static library (Firebase Storage) ----------

export type LibStone = { col: number; row: number; color: string };

export type LibProblem = {
  id: string;
  collection: string;
  source_board_idx: number;
  black_to_play: boolean;
  stones: LibStone[];
  image: string | null;
};

export type LibCollection = {
  collection: string;
  slug: string;
  count: number;
  last_uploaded_at: string;
};

// ---------- Firestore ----------

export type Role = 'player' | 'teacher';

export type UserDoc = {
  uid: string;
  displayName: string;
  email: string;
  role: Role;
  playDefaults?: PlayDefaults;
  enginePrefs?: EnginePrefs;
};

/** Preferred analysis model, shared across Review and Explore. Tracked per
 * context because the native GPU backend is only offered in local dev — so the
 * choice made when it's reachable is remembered separately from the browser one. */
export type EnginePrefs = {
  localModelId?: string;    // preferred model when the native backend is reachable
  browserModelId?: string;  // preferred model when it isn't (browser-only)
};

/** The human-like opponent's settings, kept in the AI settings modal and
 * written back as they change. Your side isn't here: it's a live toggle on the
 * board, switchable mid-game. */
export type PlayDefaults = {
  rank: string;
  temperature: number;
  moveDelay: number;
  engine?: 'browser' | 'local';    // 'local' applies only when the native backend is reachable
  scoreMode: 'show' | 'hide' | 'alert';
  alertKind?: 'behind' | 'drop';   // behind = absolute deficit, drop = recent loss
  alertThreshold: number;          // behind: alert at this deficit
  dropPoints?: number;             // drop: alert after losing this many points…
  dropMoves?: number;              // …within this many moves
};

export type Move = { col: number; row: number };

/** `attempts/{attemptId}`. `submissionId === null` ⇒ still in the open batch. */
export type AttemptDoc = {
  id: string;
  studentUid: string;
  problemId: string;
  collection: string;
  moves: Move[];
  blackToPlay: boolean;
  createdAt: number;
  submissionId: string | null;
};

/** `submissions/{submissionId}`. */
export type SubmissionDoc = {
  id: string;
  studentUid: string;
  teacherUid: string;
  sentAt: number;
  acked: boolean;
};

/** `verdicts/{attemptId}`. */
export type VerdictDoc = {
  attemptId: string;
  studentUid: string;
  teacherUid: string;
  verdict: Verdict;
  comment: string;
  reviewedAt: number;
};

/** `links/{studentUid}__{teacherUid}`. */
export type LinkDoc = {
  studentUid: string;
  teacherUid: string;
  createdAt: number;
};

// ---------- games / review ----------

/** Where a game came from. Locally-played KataGo games are `go-training`;
 * games imported from Fox Weiqi are `fox`. Other sites can be added later. */
export type GameSource = 'go-training' | 'fox' | 'gogod' | 'upload';

export type GameMove = { color: Color; x: number; y: number };

/** `games/{gameId}` — a game to review. Either a locally-played KataGo game
 * (the play-vs-KataGo fields) or one imported from an external server (the
 * import fields). The SGF is the source of truth for players, result, moves. */
export type GameDoc = {
  id: string;
  ownerUid: string;
  source: GameSource;
  createdAt: number;
  sgf: string;
  // Display name. Absent on older docs — the review list falls back by source
  // (e.g. "Fox game").
  name?: string;
  // When the game was played and where, as opposed to `createdAt` (when the
  // record was added here). Absent on games that never carried either.
  date?: string;                      // SGF DT, normally YYYY-MM-DD
  event?: string;                     // SGF EV; sources imply one, see gameEvent
  // play vs KataGo and uploads — the owner's side, when known
  myColor?: Color;
  rank?: string;                      // humanSLProfile, e.g. "rank_9k"
  rankLabel?: string;                 // "9 kyu"
  temperature?: number;
  scoreAt?: Record<string, number>;   // moveCount -> lead (Black's perspective)
  moveCount?: number;
  finalScore?: number | null;         // last recorded estimate
  // imported (Fox) only — participant uids, for the account filter
  blackUid?: number;
  whiteUid?: number;
};

/** `users/{uid}/foxAccounts/{accountUid}` — a tracked Fox player and its
 * incremental-sync cursor. The Fox account uid is the document id. */
export type FoxAccountDoc = {
  uid: number;
  username: string;
  lastChessId: string;   // newest synced game; sync pulls only games newer than this
  lastSyncedAt: number;  // ms epoch of the last successful sync
  isMine?: boolean;      // one of the owner's own accounts (vs. an imported third party)
};

/** One off-mainline node of a saved variation tree (see variations.ts). The
 * mainline is rebuilt from the game SGF, so only these are persisted. */
export type SavedNode = { id: number; parent: number; move: GameMove };

/** `reviews/{reviewId}` — a user's saved variation tree for a game. Owner-only;
 * a student's and a teacher's reviews of the same game are independent objects.
 * The schema allows multiple reviews per (owner, game) though only one is
 * surfaced today. Only off-mainline nodes are stored; the analyzed-score cache
 * is recomputed on load, never persisted. */
export type ReviewDoc = {
  id: string;
  ownerUid: string;
  gameId: string;
  nodes: SavedNode[];
  createdAt: number;
  updatedAt: number;
};
