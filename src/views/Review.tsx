import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../auth';
import { deleteGame, gameOutcome, listGames } from '../data/games';
import { deleteReviewsForGame } from '../data/reviews';
import { listStudents } from '../data/links';
import { foxAvailable, listFoxAccounts } from '../data/fox';
import type { FoxAccountDoc, GameDoc, UserDoc } from '../data/model';
import { movesFromSgf, setupStonesFromSgf, sgfInfo } from '../sgf';
import { replay } from '../goRules';
import { Board } from '../Board';
import { Spinner } from '../Spinner';
import { FilterChips, type Chip } from '../FilterChips';
import { ManagePlayersModal } from './ManagePlayersModal';
import { UploadGameModal } from './UploadGameModal';
import { EditGameModal } from './EditGameModal';
import './Review.css';

const LOCAL_AI = 'local-ai';

/** The owner's player name on a marked upload (myColor set), for the name
 * filter chips. Null when unmarked — those games are always shown. */
function uploadPlayerName(g: GameDoc): string | null {
  if (g.source !== 'upload' || !g.myColor) return null;
  const info = sgfInfo(g.sgf);
  return (g.myColor === 'B' ? info.playerBlack : info.playerWhite).trim() || 'me';
}
const uploadChipKey = (name: string) => `up:${name}`;

/** The filter-chip keys a game can be matched by: its Fox participants, the
 * vs-KataGo tag, or the owner's name on a marked upload. A game with no keys
 * (e.g. an unmarked upload) carries no tag and is always shown. */
function gameChipKeys(g: GameDoc): string[] {
  if (g.source === 'go-training') return [LOCAL_AI];
  if (g.source === 'upload') {
    const name = uploadPlayerName(g);
    return name ? [uploadChipKey(name)] : [];
  }
  return [g.blackUid, g.whiteUid].filter((u): u is number => u != null).map(String);
}

/** The filter chips one owner's games offer: their tracked Fox accounts, the
 * vs-KataGo tag when they have such a game, and the owner's name on each marked
 * upload. Opponents are deliberately absent — a game is filtered by whose game
 * it is, not by who it was against. */
function chipsFor(games: GameDoc[], accounts: FoxAccountDoc[]): Chip[] {
  const list: Chip[] = accounts.map((a) => ({ key: String(a.uid), label: a.username }));
  if (games.some((g) => g.source === 'go-training')) {
    list.push({ key: LOCAL_AI, label: 'vs KataGo' });
  }
  const names = new Set<string>();
  for (const g of games) {
    const n = uploadPlayerName(g);
    if (n) names.add(n);
  }
  for (const n of [...names].sort()) list.push({ key: uploadChipKey(n), label: n });
  return list;
}

/** Show everything: every chip on offer starts selected. */
function defaultSelection(games: GameDoc[], accounts: FoxAccountDoc[]): Set<string> {
  return new Set(chipsFor(games, accounts).map((c) => c.key));
}

/** Games one owner shared — the teacher view's scope. */
const gamesOf = (games: GameDoc[], ownerUid: string) => games.filter((g) => g.ownerUid === ownerUid);

/** Download a game's SGF, injecting the display name as GN when the stored
 * SGF lacks one — the file carries all metadata for use elsewhere. */
function downloadSgf(game: GameDoc): void {
  let sgf = game.sgf;
  if (game.name && !/\bGN\[/.test(sgf)) {
    sgf = sgf.replace('(;', `(;GN[${game.name.replace(/([\]\\])/g, '\\$1')}]`);
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([sgf], { type: 'application/x-go-sgf' }));
  a.download = `${(game.name ?? '').replace(/[^\w\- ]+/g, '').trim() || game.id}.sgf`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Display name for a game card: the stored name, else a source default. */
function gameName(g: GameDoc): string {
  if (g.name) return g.name;
  if (g.source === 'fox') return 'Fox game';
  if (g.source === 'go-training') return 'vs KataGo';
  return 'Game';
}
const PAGE_SIZE = 32;
const dedupeById = (gs: GameDoc[]) => Array.from(new Map(gs.map((g) => [g.id, g])).values());
const dedupeByUid = (as: FoxAccountDoc[]) => Array.from(new Map(as.map((a) => [a.uid, a])).values());
const scoreLabel = (lead: number) => `${lead >= 0 ? 'B' : 'W'}+${Math.abs(lead).toFixed(1)}`;
const shortDate = (ms: number) =>
  new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

/** Game result: from the SGF (imported games) or the KataGo score estimate. */
function resultLabel(g: GameDoc): string {
  const re = sgfInfo(g.sgf).result;
  if (re) return re;
  if (g.finalScore != null) return scoreLabel(g.finalScore);
  return '—';
}

/** One game as a card: the position at move 30 (or the final position if the
 * game was shorter) plus players, date, moves, and result. A missing `onDelete`
 * (e.g. a student's shared game) renders the card without the delete control.
 * `outcome` (win/loss for one of the viewer's own accounts) tints the border
 * and result. */
function GameCard({ game, outcome, onOpen, onDelete, onEdit }: {
  game: GameDoc;
  outcome: 'win' | 'loss' | null;
  onOpen: () => void;
  onDelete?: () => void;
  onEdit?: () => void;
}) {
  const moves = useMemo(() => movesFromSgf(game.sgf), [game.sgf]);
  const stones = useMemo(
    () => replay(moves.slice(0, 30), setupStonesFromSgf(game.sgf)).stones,
    [moves, game.sgf],
  );
  const info = sgfInfo(game.sgf);
  return (
    <div
      className={`game-card${outcome ? ` game-card--${outcome}` : ''}`}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter') onOpen(); }}
    >
      {onDelete && (
        <button
          type="button"
          className="game-card-del"
          aria-label="Delete game"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
        >
          ×
        </button>
      )}
      <button
        type="button"
        className="game-card-del game-card-dl"
        aria-label="Download SGF"
        title="Download SGF"
        onClick={(e) => { e.stopPropagation(); downloadSgf(game); }}
      >
        ⤓
      </button>
      {onEdit && (
        <button
          type="button"
          className="game-card-del game-card-edit"
          aria-label="Edit game details"
          title="Edit game details"
          onClick={(e) => { e.stopPropagation(); onEdit(); }}
        >
          ✎
        </button>
      )}
      <div className="game-card-board">
        <Board stones={stones} displayOnly thumbnail />
      </div>
      <div className="game-card-meta">
        <div className="game-card-title">
          <span className="game-card-name">{gameName(game)}</span>
          <span className="game-card-date">{shortDate(game.createdAt)}</span>
        </div>
        <div className="game-card-players">
          <span className="gcp-name">{info.playerBlack || 'Black'}</span>
          {info.rankBlack && <span className="review-rank">[{info.rankBlack}]</span>}
          <span className="gcp-vs">vs</span>
          <span className="gcp-name">{info.playerWhite || 'White'}</span>
          {info.rankWhite && <span className="review-rank">[{info.rankWhite}]</span>}
        </div>
        <div className="game-card-sub">
          {moves.length} moves ·{' '}
          <span className={outcome ? `game-result game-result--${outcome}` : undefined}>
            {resultLabel(game)}
          </span>
        </div>
      </div>
    </div>
  );
}

/** Game review browser. In student view it's your own games, deletable. In
 * teacher view it's one student's shared games at a time, read-only, picked
 * from a student chip row. Either way the list is scoped to a single owner and
 * filtered by that owner's player-name chips (Fox accounts, vs-KataGo, upload
 * names). */
export function Review({ teacherMode = false }: { teacherMode?: boolean }) {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [games, setGames] = useState<GameDoc[] | null>(null);
  // Tracked Fox accounts per owner: just the viewer in student view, one entry
  // per linked student in teacher view.
  const [accountsByOwner, setAccountsByOwner] = useState<Record<string, FoxAccountDoc[]>>({});
  // Fox uids of accounts the viewer owns — their own accounts (student view),
  // or, in teacher view, the "my account" accounts of every linked student.
  const [myUids, setMyUids] = useState<Set<number>>(new Set());
  const [students, setStudents] = useState<UserDoc[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Teacher view only: the one student whose games are on screen.
  const [student, setStudent] = useState<string | null>(null);
  const [foxOk, setFoxOk] = useState(false);
  const [managing, setManaging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [editing, setEditing] = useState<GameDoc | null>(null);
  // Page lives in the URL so opening a game and coming back (button or browser
  // back) returns to the same page.
  const [searchParams, setSearchParams] = useSearchParams();
  const page = Math.max(0, Number(searchParams.get('page')) || 0);
  const setPage = (p: number) =>
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (p <= 0) next.delete('page'); else next.set('page', String(p));
      return next;
    }, { replace: true });
  const backTo = searchParams.toString() ? `/review?${searchParams}` : '/review';

  useEffect(() => {
    if (!user) return;
    const uid = user.uid;
    let active = true;
    if (teacherMode) {
      listStudents(uid)
        .then(async (ss) => {
          const [gameLists, accountLists] = await Promise.all([
            Promise.all(ss.map((s) => listGames(s.uid).catch(() => []))),
            Promise.all(ss.map((s) => listFoxAccounts(s.uid).catch(() => []))),
          ]);
          if (!active) return;
          const all = dedupeById(gameLists.flat());
          const byOwner = Object.fromEntries(ss.map((s, i) => [s.uid, dedupeByUid(accountLists[i])]));
          const first = ss[0]?.uid ?? null;
          setStudents(ss);
          setAccountsByOwner(byOwner);
          setGames(all);
          setMyUids(new Set(accountLists.flat().filter((a) => a.isMine).map((a) => a.uid)));
          setStudent(first);
          setSelected(first ? defaultSelection(gamesOf(all, first), byOwner[first]) : new Set());
        })
        .catch(() => { if (active) setGames([]); });
    } else {
      Promise.all([listGames(uid), listFoxAccounts(uid)])
        .then(([own, a]) => {
          if (!active) return;
          setStudents([]);
          setAccountsByOwner({ [uid]: a });
          setGames(own);
          setMyUids(new Set(a.filter((x) => x.isMine).map((x) => x.uid)));
          setSelected(defaultSelection(own, a));
        })
        .catch(() => { if (active) setGames([]); });
      foxAvailable().then((ok) => { if (active) setFoxOk(ok); });
    }
    return () => { active = false; };
  }, [user, teacherMode]);

  // Everything below is scoped to one owner: the viewer, or the student whose
  // chip is active. Teacher and student view then behave identically.
  const scopeUid = teacherMode ? student : user?.uid ?? null;
  const accounts = useMemo(
    () => (scopeUid ? accountsByOwner[scopeUid] ?? [] : []),
    [accountsByOwner, scopeUid],
  );
  const scoped = useMemo(
    () => (scopeUid ? gamesOf(games ?? [], scopeUid) : []),
    [games, scopeUid],
  );

  // Player-name chips: Fox accounts + vs-KataGo + upload names. A teacher sees
  // the selected student's names — the same ones that student sees.
  const chips = useMemo(() => chipsFor(scoped, accounts), [scoped, accounts]);
  const chipKeys = useMemo(() => new Set(chips.map((c) => c.key)), [chips]);

  const studentChips = useMemo(
    () => (teacherMode ? students.map((s) => ({ key: s.uid, label: s.displayName })) : []),
    [teacherMode, students],
  );

  // Only keys with a chip behind them can filter: a Fox opponent nobody tracks
  // has no chip to deselect, so counting it would keep the game visible always.
  const visible = useMemo(
    () => scoped
      .filter((g) => {
        const keys = gameChipKeys(g).filter((k) => chipKeys.has(k));
        return keys.length === 0 || keys.some((k) => selected.has(k));
      })
      .sort((a, b) => b.createdAt - a.createdAt),
    [scoped, selected, chipKeys],
  );

  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const paged = visible.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);

  const toggle = (key: string) => {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
    setPage(0);   // a filter change jumps back to the first page
  };

  // One student at a time: picking another swaps in their games and their chips,
  // all selected.
  const pickStudent = (uid: string) => {
    setStudent(uid);
    setSelected(defaultSelection(gamesOf(games ?? [], uid), accountsByOwner[uid] ?? []));
    setPage(0);
  };

  // Reload after a Manage-players change (student view only); auto-select any
  // newly-added player.
  const reload = async () => {
    if (!user) return;
    const [own, a] = await Promise.all([listGames(user.uid), listFoxAccounts(user.uid)]);
    setGames(own);
    setAccountsByOwner({ [user.uid]: a });
    setMyUids(new Set(a.filter((x) => x.isMine).map((x) => x.uid)));
    setSelected((s) => new Set([...s, ...a.map((x) => String(x.uid))]));
  };

  const remove = async (id: string) => {
    if (!window.confirm('Delete this game?')) return;
    await deleteGame(id);
    if (user) await deleteReviewsForGame(user.uid, id);
    setGames((gs) => (gs ? gs.filter((g) => g.id !== id) : gs));
  };

  if (games === null) return <div className="center-screen"><Spinner /></div>;

  const studentName = students.find((s) => s.uid === student)?.displayName ?? 'this student';
  const emptyMessage = teacherMode
    ? (students.length === 0 ? 'No linked students yet.'
        : scoped.length === 0 ? `No games shared by ${studentName}.`
          : 'No games match the selected filters.')
    : (chips.length === 0
        ? (foxOk ? 'Add a player with “Manage players” to import games.' : 'No games yet.')
        : 'No games match the selected filters.');

  return (
    <div className="review">
      <div className="review-head">
        <h1>{teacherMode ? 'Shared games' : 'Games'}</h1>
        {!teacherMode && (
          <div className="review-head-actions">
            <button type="button" className="review-manage" onClick={() => setUploading(true)}>
              Upload game
            </button>
            {foxOk && (
              <button type="button" className="review-manage" onClick={() => setManaging(true)}>
                Manage players
              </button>
            )}
          </div>
        )}
      </div>

      {studentChips.length > 0 && (
        <FilterChips chips={studentChips} selected={student ? new Set([student]) : new Set()}
          onToggle={pickStudent} label="Student" />
      )}
      <FilterChips chips={chips} selected={selected} onToggle={toggle} label="Filter games" />

      {visible.length === 0 ? (
        <p className="review-empty">{emptyMessage}</p>
      ) : (
        <>
        <div className="review-grid">
          {paged.map((g) => (
            <GameCard
              key={g.id}
              game={g}
              outcome={gameOutcome(g, myUids)}
              onOpen={() => navigate(`/review/${g.id}`, { state: { from: backTo } })}
              onDelete={teacherMode ? undefined : () => remove(g.id)}
              onEdit={teacherMode ? undefined : () => setEditing(g)}
            />
          ))}
        </div>
        {visible.length > PAGE_SIZE && (
          <div className="review-pager">
            <button type="button" onClick={() => setPage(safePage - 1)} disabled={safePage === 0}>
              ← Prev
            </button>
            <span className="review-muted">
              {safePage * PAGE_SIZE + 1}–{Math.min(visible.length, (safePage + 1) * PAGE_SIZE)} of {visible.length}
            </span>
            <button type="button" onClick={() => setPage(safePage + 1)} disabled={safePage >= pageCount - 1}>
              Next →
            </button>
          </div>
        )}
        </>
      )}

      {editing && (
        <EditGameModal
          game={editing}
          onClose={() => setEditing(null)}
          onSaved={(g) => {
            setEditing(null);
            setGames((gs) => (gs ?? []).map((x) => (x.id === g.id ? g : x)));
          }}
        />
      )}
      {uploading && user && (
        <UploadGameModal
          ownerUid={user.uid}
          onClose={() => setUploading(false)}
          onSaved={(g) => {
            setUploading(false);
            setGames((gs) => dedupeById([g, ...(gs ?? [])]));
            const n = uploadPlayerName(g);
            if (n) setSelected((sel) => new Set([...sel, uploadChipKey(n)]));
            navigate(`/review/${g.id}`, { state: { from: backTo } });
          }}
        />
      )}
      {managing && user && (
        <ManagePlayersModal
          ownerUid={user.uid}
          accounts={accounts}
          games={games}
          onClose={() => setManaging(false)}
          onChanged={reload}
        />
      )}
    </div>
  );
}
