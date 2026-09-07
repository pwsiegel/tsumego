import { useState } from 'react';
import { deleteField } from 'firebase/firestore';
import { shareLink, updateGame } from '../data/games';
import type { GameDoc } from '../data/model';
import { useBackdropDismiss } from '../backdrop';
import './UploadGameModal.css';

/** Turn a game's public link on or off and copy it. Sharing exposes the game
 * record alone — saved variations live in an owner-only collection and stay
 * private, which the security rules enforce rather than this dialog. */
export function ShareGameModal({ game, onClose, onChanged }: {
  game: GameDoc;
  onClose: () => void;
  onChanged: (game: GameDoc) => void;
}) {
  const [isShared, setIsShared] = useState(!!game.shared);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const link = shareLink(game.id);

  const setSharing = async (next: boolean) => {
    setBusy(true);
    setError('');
    try {
      await updateGame(game.id, { shared: next ? true : deleteField() });
      setIsShared(next);
      onChanged({ ...game, shared: next || undefined });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not change sharing.');
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError('Could not reach the clipboard — select the link and copy it by hand.');
    }
  };

  return (
    <div className="review-modal-backdrop" {...useBackdropDismiss(onClose)} role="presentation">
      <div className="review-modal share-modal" role="dialog" aria-modal="true" aria-label="Share game">
        <button type="button" className="review-modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>Share game</h2>

        <label className="share-toggle">
          <input
            type="checkbox"
            checked={isShared}
            disabled={busy}
            onChange={(e) => setSharing(e.target.checked)}
          />
          <span>
            <strong>Anyone with the link can view this game</strong>
            <span className="share-note">
              No account needed. They see the game, the moves and the result — not your saved
              variations. Unticking this breaks the link for everyone.
            </span>
          </span>
        </label>

        <label className="upload-field">
          <span>Link</span>
          <input
            value={isShared ? link : ''}
            placeholder={isShared ? undefined : 'Turn sharing on to get a link'}
            readOnly
            disabled={!isShared}
            onFocus={(e) => e.currentTarget.select()}
          />
        </label>

        {error && <p className="review-error">{error}</p>}

        <div className="upload-actions">
          <button type="button" onClick={copy} disabled={!isShared || busy}>
            {copied ? 'Copied' : 'Copy link'}
          </button>
        </div>
      </div>
    </div>
  );
}
