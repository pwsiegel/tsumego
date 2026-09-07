import { useMemo, useRef, useState } from 'react';
import { saveGame } from '../data/games';
import type { GameDoc } from '../data/model';
import type { Color } from '../types';
import { mainlineMovesFromSgf, setupStonesFromSgf, sgfInfo, standardHandicapStones, toSgf } from '../sgf';
import { useBackdropDismiss } from '../backdrop';
import './UploadGameModal.css';

/** Paste (or pick a file of) SGF, adjust the metadata, save it as an
 * uploaded game for review. The metadata fields auto-populate from the SGF's
 * tags when present and stay blank when not; the stored SGF is regenerated as
 * a clean main line carrying the (possibly edited) metadata. */
export function UploadGameModal({ ownerUid, onClose, onSaved }: {
  ownerUid: string;
  onClose: () => void;
  onSaved: (game: GameDoc) => void;
}) {
  const [sgfText, setSgfText] = useState('');
  const [name, setName] = useState('');
  const [date, setDate] = useState('');
  const [event, setEvent] = useState('');
  const [playerBlack, setPlayerBlack] = useState('');
  const [rankBlack, setRankBlack] = useState('');
  const [playerWhite, setPlayerWhite] = useState('');
  const [rankWhite, setRankWhite] = useState('');
  const [winner, setWinner] = useState<'' | Color>('');
  const [scoreText, setScoreText] = useState('');
  const [rulesText, setRulesText] = useState('');
  const [komiText, setKomiText] = useState('');
  const [handicap, setHandicap] = useState(0);
  const [myColor, setMyColor] = useState<Color | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const parsed = useMemo(() => {
    if (!sgfText.trim()) return null;
    return {
      info: sgfInfo(sgfText),
      moves: mainlineMovesFromSgf(sgfText),
      setup: setupStonesFromSgf(sgfText),
    };
  }, [sgfText]);

  const applySgf = (text: string) => {
    setSgfText(text);
    setError('');
    if (!text.trim()) return;
    const info = sgfInfo(text);
    setName(info.name);
    setDate(info.date);
    setEvent(info.event);
    setPlayerBlack(info.playerBlack);
    setRankBlack(info.rankBlack);
    setPlayerWhite(info.playerWhite);
    setRankWhite(info.rankWhite);
    const re = info.result.match(/^([BW])\+(.*)$/);
    setWinner(re ? (re[1] as Color) : '');
    setScoreText(re ? re[2] : '');
    setRulesText(info.rules);
    setKomiText(info.komi != null ? String(info.komi) : '');
    const abCount = setupStonesFromSgf(text).filter((st) => st.color === 'B').length;
    setHandicap(info.handicap ?? (abCount >= 2 ? abCount : 0));
  };

  const pickFile = async (file: File | undefined) => {
    if (!file) return;
    applySgf(await file.text());
  };

  const problem = !parsed ? null
    : parsed.moves.length === 0 ? 'No moves found — is this an SGF game record?'
      : parsed.info.boardSize != null && parsed.info.boardSize !== 19 ? `Only 19×19 games are supported (this is ${parsed.info.boardSize}×${parsed.info.boardSize}).`
        : null;

  const save = async () => {
    if (!parsed || problem) return;
    setSaving(true);
    setError('');
    const komiVal = komiText.trim() === '' ? null : Number(komiText);
    if (komiVal !== null && !Number.isFinite(komiVal)) {
      setSaving(false);
      setError('Komi must be a number.');
      return;
    }
    try {
      const { info, moves } = parsed;
      // Winner + score compose RE ("B+R", "W+2.5"); no winner keeps the SGF's
      // own result. Setup stones: the SGF's when they match the chosen
      // handicap, else the standard hoshi placement.
      const score = scoreText.trim().replace(/^r(es(ign)?)?$/i, 'R');
      const result = winner ? `${winner}+${score}` : info.result;
      const abCount = parsed.setup.filter((st) => st.color === 'B').length;
      const setup = handicap >= 2
        ? (abCount === handicap ? parsed.setup : standardHandicapStones(handicap))
        : parsed.setup;
      const sgf = toSgf(moves, {
        komi: komiVal,
        rules: rulesText.trim(),
        handicap,
        setup,
        name: name.trim(),
        playerBlack: playerBlack.trim(),
        playerWhite: playerWhite.trim(),
        rankBlack: rankBlack.trim(),
        rankWhite: rankWhite.trim(),
        date: date.trim(),
        event: event.trim(),
        result,
      });
      const game = await saveGame({
        ownerUid,
        source: 'upload',
        createdAt: Date.now(),
        sgf,
        ...(name.trim() ? { name: name.trim() } : {}),
        ...(date.trim() ? { date: date.trim() } : {}),
        ...(event.trim() ? { event: event.trim() } : {}),
        ...(myColor ? { myColor } : {}),
      });
      onSaved(game);
    } catch (e) {
      setSaving(false);
      setError(e instanceof Error ? e.message : 'Could not save the game.');
    }
  };

  return (
    <div className="review-modal-backdrop" {...useBackdropDismiss(onClose)} role="presentation">
      <div
        className="review-modal upload-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Upload game"
      >
        <button type="button" className="review-modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>Upload game</h2>

        <textarea
          className="upload-sgf"
          placeholder="Paste SGF here…"
          value={sgfText}
          onChange={(e) => applySgf(e.target.value)}
          spellCheck={false}
        />
        <div className="upload-file-row">
          <button type="button" onClick={() => fileRef.current?.click()}>Choose .sgf file…</button>
          <input
            ref={fileRef}
            type="file"
            accept=".sgf"
            style={{ display: 'none' }}
            onChange={(e) => { void pickFile(e.target.files?.[0]); e.target.value = ''; }}
          />
          {parsed && !problem && (
            <span className="upload-parsed">
              {parsed.moves.length} moves{parsed.info.result && ` · ${parsed.info.result}`}
              {parsed.info.date && ` · ${parsed.info.date}`}
            </span>
          )}
        </div>

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
          <label className="upload-field">
            <span>Handicap</span>
            <select value={handicap} onChange={(e) => setHandicap(Number(e.target.value))}>
              <option value={0}>None</option>
              {[2, 3, 4, 5, 6, 7, 8, 9].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </div>

        <div className="upload-me-row">
          <span className="upload-me-label">You played</span>
          {([['B', 'Black'], ['W', 'White'], [null, 'Neither']] as [Color | null, string][]).map(([v, label]) => (
            <label key={label} className="upload-me-option">
              <input
                type="radio"
                name="upload-me"
                checked={myColor === v}
                onChange={() => setMyColor(v)}
              />
              {label}
            </label>
          ))}
        </div>

        {problem && <p className="review-error">{problem}</p>}
        {error && <p className="review-error">{error}</p>}

        <div className="upload-actions">
          <button type="button" onClick={save} disabled={saving || !parsed || !!problem}>
            {saving ? 'Saving…' : 'Save game'}
          </button>
        </div>
      </div>
    </div>
  );
}
