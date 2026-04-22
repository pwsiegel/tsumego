import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

type Collection = {
  source: string;
  count: number;
  accepted: number;
  accepted_edited: number;
  rejected: number;
  unreviewed: number;
  last_uploaded_at: string;
};

function formatDate(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

async function fetchCollections(): Promise<Collection[]> {
  const r = await fetch('/api/tsumego/collections', { cache: 'no-store' });
  if (!r.ok) throw new Error(r.statusText);
  return (await r.json()).collections;
}

export function Home() {
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);   // source being deleted

  useEffect(() => {
    fetchCollections()
      .then(setCollections)
      .catch((e) => { setError(String(e)); setCollections([]); });
  }, []);

  const deleteCollection = async (c: Collection) => {
    const ok = confirm(
      `Delete ${c.count} problem${c.count === 1 ? '' : 's'} from “${c.source}”?`
    );
    if (!ok) return;
    setBusy(c.source);
    try {
      const r = await fetch(
        `/api/tsumego/collections/${encodeURIComponent(c.source)}`,
        { method: 'DELETE' },
      );
      if (!r.ok) throw new Error(r.statusText);
      const next = await fetchCollections();
      setCollections(next);
    } catch (e) {
      setError(`Delete failed: ${e}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="home">
      <header className="home-header">
        <h1>Go problem workbook</h1>
        <nav>
          <Link to="/testing" className="dim">developer tools</Link>
        </nav>
      </header>

      <section className="home-actions">
        <Link to="/upload" className="primary-btn">
          Upload a PDF
        </Link>
        <Link to="/tsumego" className="secondary-btn">
          Solve puzzles
        </Link>
      </section>

      <section className="collections">
        <h2>Collections</h2>
        {collections === null && <p className="dim">Loading…</p>}
        {collections !== null && collections.length === 0 && (
          <p className="dim">
            No collections yet. Upload a PDF to import problems.
          </p>
        )}
        {collections !== null && collections.length > 0 && (
          <ul className="collection-list">
            {collections.map((c) => (
              <li key={c.source} className="collection-item">
                <Link
                  to={`/collections/${encodeURIComponent(c.source)}`}
                  className="collection-link"
                >
                  <div className="collection-info">
                    <div className="collection-source">{c.source}</div>
                    <div className="collection-meta">
                      {c.count} {c.count === 1 ? 'problem' : 'problems'}
                      {c.unreviewed > 0 && (
                        <> &nbsp;·&nbsp; <span className="stat-unreviewed">{c.unreviewed} unreviewed</span></>
                      )}
                      {c.accepted > 0 && (
                        <> &nbsp;·&nbsp; <span className="stat-accepted">{c.accepted} accepted</span></>
                      )}
                      {c.accepted_edited > 0 && (
                        <> &nbsp;·&nbsp; <span className="stat-edited">{c.accepted_edited} edited</span></>
                      )}
                      {c.rejected > 0 && (
                        <> &nbsp;·&nbsp; <span className="stat-rejected">{c.rejected} rejected</span></>
                      )}
                      {c.last_uploaded_at && (
                        <>
                          <span className="dot">·</span>
                          <span>added {formatDate(c.last_uploaded_at)}</span>
                        </>
                      )}
                    </div>
                  </div>
                </Link>
                <button
                  className="collection-delete"
                  onClick={() => deleteCollection(c)}
                  disabled={busy === c.source}
                  aria-label={`Delete collection ${c.source}`}
                  title="Delete collection"
                >
                  <svg
                    width="16" height="16" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  >
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                    <path d="M10 11v6" />
                    <path d="M14 11v6" />
                    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
        {error && <p className="error">{error}</p>}
      </section>
    </div>
  );
}
