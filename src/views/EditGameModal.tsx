import { useState } from 'react';
import { deleteField } from 'firebase/firestore';
import { updateGame } from '../data/games';
import type { GameDoc } from '../data/model';
import type { Color } from '../types';
import { patchSgfMeta, sgfInfo } from '../sgf';
import './UploadGameModal.css';

/** Edit a game's metadata — name, date, event, players/ranks, result, ruleset, komi,
 * and the you-played marker. Edits patch the SGF's root tags in place, so the
 * move record (including passes and variations) is untouched; the board
 * position and moves are not editable here. */
export function EditGameModal({ game, onClose, onSaved }: {
  game: GameDoc;
  onClose: () => void;
  onSaved: (game: GameDoc) => void;
}) {
  const info = sgfInfo(game.sgf);
  const re = info.result.match(/^([BW])\+(.*)$/);
  const [name, setName] = useState(game.name ?? '');
  const [date, setDate] = useState(info.date);
  const [event, setEvent] = useState(info.event);
  const [playerBlack, setPlayerBlack] = useState(info.playerBlack);
  const [rankBlack, setRankBlack] = useState(info.rankBlack);
  const [playerWhite, setPlayerWhite] = useState(info.playerWhite);
  const [rankWhite, setRankWhite] = useState(info.rankWhite);
  const [winner, setWinner] = useState<'' | Color>(re ? (re[1] as Color) : '');
  const [scoreText, setScoreText] = useState(re ? re[2] : '');
  const [rulesText, setRulesText] = useState(info.rules);
  const [komiText, setKomiText] = useState(info.komi != null ? String(info.komi) : '');
  const [myColor, setMyColor] = useState<Color | null>(game.myColor ?? null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const save = async () => {
    setSaving(true);
    setError('');
    const komi = komiText.trim();
    if (komi !== '' && !Number.isFinite(Number(komi))) {
      setSaving(false);
      setError('Komi must be a number.');
      return;
    }
    try {
      const score = scoreText.trim().replace(/^r(es(ign)?)?$/i, 'R');
      const sgf = patchSgfMeta(game.sgf, {
        GN: name.trim(),
        DT: date.trim(),
        EV: event.trim(),
        PB: playerBlack.trim(),
        BR: rankBlack.trim(),
        PW: playerWhite.trim(),
        WR: rankWhite.trim(),
        // No winner selected leaves the SGF's result as it was.
        ...(winner ? { RE: `${winner}+${score}` } : {}),
        KM: komi,
        RU: rulesText.trim(),
      });
      const patch = {
        sgf,
        name: name.trim() ? name.trim() : deleteField(),
        date: date.trim() ? date.trim() : deleteField(),
        event: event.trim() ? event.trim() : deleteField(),
        myColor: myColor ?? deleteField(),
      };
      await updateGame(game.id, patch);
      const updated: GameDoc = {
        ...game,
        sgf,
        name: name.trim() || undefined,
        date: date.trim() || undefined,
        event: event.trim() || undefined,
        myColor: myColor ?? undefined,
      };
      onSaved(updated);
    } catch (e) {
      setSaving(false);
      setError(e instanceof Error ? e.message : 'Could not save.');
    }
  };

  return (
    <div className="review-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="review-modal upload-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Edit game"
        onClick={(e) => e.stopPropagation()}
      >
        <button type="button" className="review-modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>Edit game</h2>

        <label className="upload-field">
          <span>Game name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="upload-player-row">
          <label className="upload-field">
            <span>Event</span>
            <input value={event} onChange={(e) => setEvent(e.target.value)} placeholder="e.g. club league" />
          </label>
          <label className="upload-field upload-field-rank">
            <span>Date</span>
            <input value={date} onChange={(e) => setDate(e.target.value)} placeholder="YYYY-MM-DD" />
          </label>
        </div>
        <div className="upload-player-row">
          <label className="upload-field">
            <span>Black player</span>
            <input value={playerBlack} onChange={(e) => setPlayerBlack(e.target.value)} />
          </label>
          <label className="upload-field upload-field-rank">
            <span>Rank</span>
            <input value={rankBlack} onChange={(e) => setRankBlack(e.target.value)} placeholder="e.g. 5k" />
          </label>
        </div>
        <div className="upload-player-row">
          <label className="upload-field">
            <span>White player</span>
            <input value={playerWhite} onChange={(e) => setPlayerWhite(e.target.value)} />
          </label>
          <label className="upload-field upload-field-rank">
            <span>Rank</span>
            <input value={rankWhite} onChange={(e) => setRankWhite(e.target.value)} placeholder="e.g. 5k" />
          </label>
        </div>
        <div className="upload-player-row">
          <label className="upload-field">
            <span>Who won</span>
            <select value={winner} onChange={(e) => setWinner(e.target.value as '' | Color)}>
              <option value="">—</option>
              <option value="B">Black</option>
              <option value="W">White</option>
            </select>
          </label>
          <label className="upload-field">
            <span>Score</span>
            <input value={scoreText} onChange={(e) => setScoreText(e.target.value)} placeholder="e.g. 2.5 or R" />
          </label>
        </div>
        <div className="upload-player-row">
          <label className="upload-field">
            <span>Ruleset</span>
            <input value={rulesText} onChange={(e) => setRulesText(e.target.value)} placeholder="e.g. AGA" />
          </label>
          <label className="upload-field">
            <span>Komi</span>
            <input value={komiText} onChange={(e) => setKomiText(e.target.value)} placeholder="e.g. 7.5" />
          </label>
        </div>

        <div className="upload-me-row">
          <span className="upload-me-label">You played</span>
          {([['B', 'Black'], ['W', 'White'], [null, 'Neither']] as [Color | null, string][]).map(([v, label]) => (
            <label key={label} className="upload-me-option">
              <input
                type="radio"
                name="edit-me"
                checked={myColor === v}
                onChange={() => setMyColor(v)}
              />
              {label}
            </label>
          ))}
        </div>

        {error && <p className="review-error">{error}</p>}

        <div className="upload-actions">
          <button type="button" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save changes'}
          </button>
        </div>
      </div>
    </div>
  );
}
