import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from './api';
import './Profile.css';

/** Settings page. Currently only display_name (the label that shows up
 * to teachers in place of the opaque user_id). Designed to grow — extra
 * fields drop in as additional rows. */
type Role = 'student' | 'teacher' | '';

export function Profile() {
  const [displayName, setDisplayName] = useState<string>('');
  const [defaultRole, setDefaultRole] = useState<Role>('');
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    api.study.getProfile()
      .then((p) => {
        setDisplayName(p.display_name ?? '');
        setDefaultRole((p.default_role as Role | null) ?? '');
        setLoaded(true);
      })
      .catch((e) => { setError(String(e)); setLoaded(true); });
  }, []);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setFlash(null);
    try {
      const trimmed = displayName.trim();
      const updated = await api.study.updateProfile({
        display_name: trimmed === '' ? null : trimmed,
        default_role: defaultRole === '' ? null : defaultRole,
      });
      setDisplayName(updated.display_name ?? '');
      setDefaultRole((updated.default_role as Role | null) ?? '');
      setFlash('Saved.');
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="profile">
      <header className="profile-header">
        <Link to="/" className="back-link">← home</Link>
        <h1>Profile</h1>
      </header>

      {!loaded ? (
        <p className="dim">Loading…</p>
      ) : (
        <form className="profile-form" onSubmit={save}>
          <label className="profile-row">
            <span className="profile-label">Display name</span>
            <input
              type="text"
              className="profile-input"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Your name (shown on submissions to teachers)"
              maxLength={64}
            />
          </label>
          <p className="profile-hint">
            How you'll appear to teachers when you send them problems.
            Leave empty to fall back to your account ID.
          </p>

          <label className="profile-row">
            <span className="profile-label">Default view</span>
            <select
              className="profile-input"
              value={defaultRole}
              onChange={(e) => setDefaultRole(e.target.value as Role)}
            >
              <option value="">Student (default)</option>
              <option value="student">Student</option>
              <option value="teacher">Teacher</option>
            </select>
          </label>
          <p className="profile-hint">
            Where the app sends you when you load <code>/</code>. You can
            still navigate to the other view from inside the app.
          </p>

          <div className="profile-actions">
            <button type="submit" className="profile-save" disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </button>
            {flash && <span className="profile-flash">{flash}</span>}
            {error && <span className="profile-error">{error}</span>}
          </div>
        </form>
      )}
    </div>
  );
}
