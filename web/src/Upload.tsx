import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from './api';
import './Upload.css';

export function Upload() {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [uploadFrac, setUploadFrac] = useState<number | null>(null);
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'rendering' | 'detecting' | 'done' | 'error'>('idle');
  const [totalPages, setTotalPages] = useState<number | null>(null);
  const [pagesRendered, setPagesRendered] = useState(0);
  const [boardsSaved, setBoardsSaved] = useState(0);
  const [source, setSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    setPhase('uploading');
    setUploadFrac(0);
    setTotalPages(null);
    setPagesRendered(0);
    setBoardsSaved(0);
    setError(null);
    setSource(null);

    try {
      await api.pdf.streamIngest(file, (frac) => setUploadFrac(frac), (ev) => {
        if (ev.event === 'start') {
          setTotalPages(ev.total_pages);
          setSource(ev.source);
          setPhase('rendering');
          setUploadFrac(1);
        } else if (ev.event === 'page_rendered') {
          setPagesRendered(ev.page);
          if (ev.page >= ev.total_pages) setPhase('detecting');
        } else if (ev.event === 'board_saved') {
          setBoardsSaved(ev.total_saved);
        } else if (ev.event === 'done') {
          setPhase('done');
          setBoardsSaved(ev.total_saved);
          setSource(ev.source);
          // Auto-navigate to the collection after a short beat so the
          // user sees the final count.
          setTimeout(() => {
            navigate(`/collections/${encodeURIComponent(ev.source)}`);
          }, 1200);
        } else if (ev.event === 'error') {
          setError(ev.detail);
          setPhase('error');
        }
      });
    } catch (e) {
      setError(String(e));
      setPhase('error');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <header className="upload-header">
        <h1>Upload a PDF</h1>
        <nav><Link to="/">home</Link></nav>
      </header>

      <div className="upload-landing">
        {phase === 'idle' && (
          <>
            <label className="upload-btn-big">
              Upload a PDF
              <input
                type="file"
                accept="application/pdf"
                style={{ display: 'none' }}
                disabled={uploading}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                  e.target.value = '';
                }}
              />
            </label>
            <p className="upload-hint">
              We'll detect every board and save them as unreviewed. You
              can review and edit each problem afterward from the
              collection page.
            </p>
          </>
        )}

        {phase !== 'idle' && phase !== 'error' && (
          <div className="ingest-progress">
            {phase === 'uploading' && uploadFrac !== null && (
              <>
                <p>Uploading… {Math.round(uploadFrac * 100)}%</p>
                <progress value={uploadFrac} max={1} className="upload-progress" />
              </>
            )}
            {phase === 'rendering' && totalPages !== null && (
              <>
                <p>Rendering pages: {pagesRendered} / {totalPages}</p>
                <progress value={pagesRendered} max={totalPages} className="upload-progress" />
              </>
            )}
            {phase === 'detecting' && (
              <>
                <p>Detecting boards &amp; saving problems… {boardsSaved} saved</p>
                <progress className="upload-progress" />
              </>
            )}
            {phase === 'done' && source && (
              <>
                <p><strong>{boardsSaved}</strong> problems saved from {source}.</p>
                <p className="dim">
                  Taking you to <Link to={`/collections/${encodeURIComponent(source)}`}>the collection</Link>…
                </p>
              </>
            )}
          </div>
        )}

        {phase === 'error' && (
          <div className="upload-error">
            <p>Upload failed: {error}</p>
            <p>
              <button onClick={() => { setPhase('idle'); setError(null); }}>
                Try again
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
