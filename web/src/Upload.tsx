import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from './api';
import './Upload.css';

export function Upload() {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [uploadFrac, setUploadFrac] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    setUploadFrac(0);
    setError(null);

    try {
      const sessionId = await api.pdf.startPatchSession(
        file, (frac) => setUploadFrac(frac),
      );
      navigate(`/upload/${sessionId}`);
    } catch (e) {
      setError(String(e));
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
        {!uploading && !error && (
          <>
            <label className="upload-btn-big">
              Upload a PDF
              <input
                type="file"
                accept="application/pdf"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                  e.target.value = '';
                }}
              />
            </label>
            <p className="upload-hint">
              We'll detect every board on every page, then walk you
              through them so you can skip bad ones and add boards we
              missed before ingesting.
            </p>
          </>
        )}

        {uploading && (
          <div className="ingest-progress">
            <p>Uploading… {Math.round((uploadFrac ?? 0) * 100)}%</p>
            <progress value={uploadFrac ?? 0} max={1} className="upload-progress" />
            <p className="dim">
              Once uploaded, board detection runs and you'll be sent
              to the bbox walker.
            </p>
          </div>
        )}

        {error && (
          <div className="upload-error">
            <p>Upload failed: {error}</p>
            <p>
              <button onClick={() => { setError(null); }}>
                Try again
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
