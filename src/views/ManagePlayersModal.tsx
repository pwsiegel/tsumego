import { useEffect, useMemo, useState } from 'react';
import { deleteFoxPlayer, onboardFoxAccount, setFoxAccountMine, syncFoxAccount } from '../data/fox';
import type { FoxAccountDoc, GameDoc } from '../data/model';
import { useBackdropDismiss } from '../backdrop';

const shortDate = (ms: number) =>
  new Date(ms).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

/** Add / sync / delete tracked Fox players. Onboarding back-fills the last 100
 * games; sync pulls only newer ones; delete removes the player and their games.
 * `onChanged` refreshes the underlying games + accounts after any change. */
export function ManagePlayersModal({
  ownerUid, accounts, games, onClose, onChanged,
}: {
  ownerUid: string;
  accounts: FoxAccountDoc[];
  games: GameDoc[];
  onClose: () => void;
  onChanged: () => Promise<void>;
}) {
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState('');    // key of the in-flight action ('' = idle)
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev; };
  }, [onClose]);

  // Games per player (fox games where the player is black or white).
  const gameCount = useMemo(() => {
    const m = new Map<number, number>();
    for (const g of games) {
      if (g.source !== 'fox') continue;
      if (g.blackUid != null) m.set(g.blackUid, (m.get(g.blackUid) ?? 0) + 1);
      if (g.whiteUid != null && g.whiteUid !== g.blackUid) m.set(g.whiteUid, (m.get(g.whiteUid) ?? 0) + 1);
    }
    return m;
  }, [games]);

  const run = async (key: string, fn: () => Promise<string>) => {
    if (busy) return;
    setBusy(key); setStatus(''); setError('');
    try {
      const msg = await fn();
      await onChanged();
      setStatus(msg);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong');
    } finally { setBusy(''); }
  };

  const add = () => {
    const name = newName.trim();
    if (!name) return;
    run('add', async () => {
      const acct = await onboardFoxAccount(ownerUid, name);
      setNewName('');
      return `Added ${acct.username}`;
    });
  };

  const sync = (a: FoxAccountDoc) =>
    run(`sync:${a.uid}`, async () => {
      const n = await syncFoxAccount(ownerUid, a);
      return `${a.username}: ${n} new game${n === 1 ? '' : 's'}`;
    });

  const toggleMine = (a: FoxAccountDoc) =>
    run(`mine:${a.uid}`, async () => {
      await setFoxAccountMine(ownerUid, a, !a.isMine);
      return a.isMine ? `${a.username} is no longer marked as yours` : `Marked ${a.username} as your account`;
    });

  const del = (a: FoxAccountDoc) => {
    if (!window.confirm(`Delete ${a.username} and their games?`)) return;
    run(`del:${a.uid}`, async () => {
      const removed = await deleteFoxPlayer(ownerUid, a);
      return `Deleted ${a.username} (${removed} game${removed === 1 ? '' : 's'} removed)`;
    });
  };

  return (
    <div className="review-modal-backdrop" {...useBackdropDismiss(onClose)} role="presentation">
      <div
        className="review-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Manage players"
      >
        <button type="button" className="review-modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>Manage players</h2>

        <div className="review-add-row">
          <input
            className="review-add"
            placeholder="Fox username"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') add(); }}
            disabled={!!busy}
          />
          <button type="button" onClick={add} disabled={!!busy || !newName.trim()}>
            {busy === 'add' ? 'Adding…' : 'Add player'}
          </button>
        </div>

        {status && <p className="review-status">{status}</p>}
        {error && <p className="review-error">{error}</p>}

        {accounts.length === 0 ? (
          <p className="review-muted">No players yet — add a Fox username to import their games.</p>
        ) : (
          <ul className="review-players">
            {accounts.map((a) => {
              const n = gameCount.get(a.uid) ?? 0;
              return (
                <li key={a.uid} className="review-player">
                  <div className="review-player-info">
                    <span className="review-player-name">{a.username}</span>
                    <span className="review-muted">
                      {n} game{n === 1 ? '' : 's'} · synced {shortDate(a.lastSyncedAt)}
                    </span>
                    <label className="review-player-mine">
                      <input
                        type="checkbox"
                        checked={!!a.isMine}
                        disabled={!!busy}
                        onChange={() => toggleMine(a)}
                      />
                      My account
                    </label>
                  </div>
                  <div className="review-player-actions">
                    <button type="button" onClick={() => sync(a)} disabled={!!busy}>
                      {busy === `sync:${a.uid}` ? 'Syncing…' : 'Sync'}
                    </button>
                    <button
                      type="button"
                      className="review-player-del"
                      onClick={() => del(a)}
                      disabled={!!busy}
                    >
                      {busy === `del:${a.uid}` ? 'Deleting…' : 'Delete'}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
