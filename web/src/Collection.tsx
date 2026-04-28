import { useEffect, useState } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { api, type TsumegoProblem } from './api';
import './Collection.css';

export function Collection() {
  const { source: encSource = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const navigate = useNavigate();
  const [problems, setProblems] = useState<TsumegoProblem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draftName, setDraftName] = useState(source);
  const [savingName, setSavingName] = useState(false);

  const refetch = async () => {
    try {
      setProblems(await api.tsumego.listProblems(source));
    } catch (e) {
      setError(String(e));
      setProblems([]);
    }
  };

  useEffect(() => {
    // refetch() owns its own state writes; the lint rule can't see through.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  const startRename = () => {
    setDraftName(source);
    setRenaming(true);
  };
  const cancelRename = () => {
    setRenaming(false);
    setDraftName(source);
  };
  const saveRename = async () => {
    const next = draftName.trim();
    if (!next || next === source) {
      cancelRename();
      return;
    }
    setSavingName(true);
    setError(null);
    try {
      await api.tsumego.renameCollection(source, next);
      setRenaming(false);
      // The URL is keyed by source — navigate to the new one. `replace`
      // keeps the back button useful.
      navigate(`/collections/${encodeURIComponent(next)}`, { replace: true });
    } catch (e) {
      setError(`Rename failed: ${e}`);
    } finally {
      setSavingName(false);
    }
  };

  const enterSelect = () => {
    setSelecting(true);
    setSelectedIds(new Set());
  };
  const exitSelect = () => {
    setSelecting(false);
    setSelectedIds(new Set());
  };
  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const deleteSelected = async () => {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    const ok = confirm(
      `Delete ${ids.length} problem${ids.length === 1 ? '' : 's'} from this collection? This cannot be undone.`
    );
    if (!ok) return;
    setDeleting(true);
    try {
      // Fire the deletes in parallel — each hits its own tsumego_*.json
      // so there's no contention.
      await Promise.all(ids.map((id) => api.tsumego.deleteProblem(id)));
      exitSelect();
      await refetch();
    } catch (e) {
      setError(`Delete failed: ${e}`);
    } finally {
      setDeleting(false);
    }
  };

  const counts = {
    unreviewed: 0, accepted: 0, accepted_edited: 0, rejected: 0,
  } as Record<string, number>;
  (problems ?? []).forEach((p) => {
    if (p.status in counts) counts[p.status]++;
  });
  const firstUnreviewed = (problems ?? []).find((p) => p.status === 'unreviewed');

  return (
    <div className="collection">
      <header className="collection-header">
        <div>
          <Link to="/" className="back-link">← home</Link>
          {renaming ? (
            <form
              className="title-edit"
              onSubmit={(e) => { e.preventDefault(); saveRename(); }}
            >
              <input
                type="text"
                className="title-edit-input"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                disabled={savingName}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Escape') cancelRename();
                }}
              />
              <button
                type="submit"
                disabled={savingName || !draftName.trim() || draftName.trim() === source}
              >
                {savingName ? 'Saving…' : 'Save'}
              </button>
              <button
                type="button"
                onClick={cancelRename}
                disabled={savingName}
                className="title-edit-cancel"
              >
                Cancel
              </button>
            </form>
          ) : (
            <h1 className="collection-title">
              {source}
              <button
                type="button"
                className="title-edit-btn"
                onClick={startRename}
                aria-label="Rename collection"
                title="Rename collection"
              >
                <svg
                  width="14" height="14" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                >
                  <path d="M12 20h9" />
                  <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
                </svg>
              </button>
            </h1>
          )}
          {problems && (
            <div className="collection-stats">
              {problems.length} total &nbsp;·&nbsp;
              <span className="stat-unreviewed">{counts.unreviewed} unreviewed</span> &nbsp;·&nbsp;
              <span className="stat-accepted">{counts.accepted} accepted</span>
              {counts.accepted_edited > 0 && (
                <> &nbsp;·&nbsp; <span className="stat-accepted_edited">{counts.accepted_edited} edited</span></>
              )}
              {counts.rejected > 0 && (
                <> &nbsp;·&nbsp; <span className="stat-rejected">{counts.rejected} rejected</span></>
              )}
            </div>
          )}
        </div>
        <div className="collection-header-actions">
          {!selecting && (
            <>
              <button
                className="review-btn"
                disabled={!firstUnreviewed}
                onClick={() => navigate(`/collections/${encodeURIComponent(source)}/review`)}
              >
                Review unreviewed ({counts.unreviewed})
              </button>
              <button
                className="review-rejected-btn"
                disabled={counts.rejected === 0}
                onClick={() => navigate(`/collections/${encodeURIComponent(source)}/review?status=rejected`)}
                title="Revisit problems you previously rejected"
              >
                Review rejected ({counts.rejected})
              </button>
              <button
                className="delete-mode-btn"
                onClick={enterSelect}
                disabled={(problems?.length ?? 0) === 0}
              >
                Delete…
              </button>
            </>
          )}
          {selecting && (
            <>
              <span className="selection-count">
                {selectedIds.size} selected
              </span>
              <button className="cancel-btn" onClick={exitSelect} disabled={deleting}>
                Cancel
              </button>
              <button
                className="confirm-delete-btn"
                onClick={deleteSelected}
                disabled={selectedIds.size === 0 || deleting}
              >
                {deleting ? 'Deleting…' : `Delete ${selectedIds.size}`}
              </button>
            </>
          )}
        </div>
      </header>

      {error && <div className="collection-error">{error}</div>}

      {problems !== null && problems.length === 0 && !error && (
        <p className="collection-empty">No problems in this collection.</p>
      )}

      {problems !== null && problems.length > 0 && (
        <ul className="problem-grid">
          {problems.map((p) => {
            const selected = selectedIds.has(p.id);
            const tileTarget = `/collections/${encodeURIComponent(source)}/problem/${p.id}`;
            const tileBody = (
              <>
                <div className="tile-thumb">
                  {p.image ? (
                    <img src={api.tsumego.imageUrl(p.id)} alt="" />
                  ) : (
                    <div className="tile-noimg">no image</div>
                  )}
                  <div className={`tile-badge badge-${p.status}`}>
                    {statusSymbol(p.status)}
                  </div>
                  {selecting && (
                    <div className="tile-checkbox" aria-label={selected ? 'selected' : 'not selected'}>
                      {selected ? '✓' : ''}
                    </div>
                  )}
                </div>
                <div className="tile-caption">
                  #{p.source_board_idx + 1}
                  <span className="tile-status">{humanStatus(p.status)}</span>
                </div>
              </>
            );
            return (
              <li
                key={p.id}
                className={`problem-tile status-${p.status}${selected ? ' selected' : ''}`}
              >
                {selecting ? (
                  <button
                    type="button"
                    className="tile-select-btn"
                    onClick={() => toggleSelect(p.id)}
                    disabled={deleting}
                    aria-pressed={selected}
                  >
                    {tileBody}
                  </button>
                ) : (
                  <Link to={tileTarget}>
                    {tileBody}
                  </Link>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function statusSymbol(s: string): string {
  if (s === 'accepted') return '✓';
  if (s === 'accepted_edited') return '✓*';
  if (s === 'rejected') return '✗';
  return '?';
}

function humanStatus(s: string): string {
  if (s === 'accepted') return 'accepted';
  if (s === 'accepted_edited') return 'edited';
  if (s === 'rejected') return 'rejected';
  return 'unreviewed';
}
