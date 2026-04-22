import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Upload.css';

type IngestEvent =
  | { event: 'start'; source: string; uploaded_at: string; total_pages: number }
  | { event: 'page_rendered'; page: number; total_pages: number }
  | { event: 'board_saved'; source_board_idx: number; page_idx: number; bbox_idx: number; total_saved: number }
  | { event: 'done'; source: string; total_saved: number; skipped: number }
  | { event: 'error'; detail: string };

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
      await streamIngest(file, (frac) => setUploadFrac(frac), (ev) => {
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


/**
 * Upload a file to /api/pdf/ingest and consume the NDJSON streaming body,
 * calling `onProgress` with the upload fraction during the request body
 * and `onEvent` for each parsed server event.
 */
function streamIngest(
  file: File,
  onProgress: (frac: number) => void,
  onEvent: (ev: IngestEvent) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/pdf/ingest');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    // XHR gives us responseText incrementally. Parse as NDJSON.
    let offset = 0;
    xhr.onprogress = () => {
      const text = xhr.responseText ?? '';
      while (true) {
        const nl = text.indexOf('\n', offset);
        if (nl === -1) break;
        const line = text.slice(offset, nl).trim();
        offset = nl + 1;
        if (!line) continue;
        try {
          onEvent(JSON.parse(line) as IngestEvent);
        } catch {
          // ignore unparseable lines
        }
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(xhr.statusText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(form);
  });
}
