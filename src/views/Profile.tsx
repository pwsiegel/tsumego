import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useAuth } from '../auth';
import { setDisplayName, setRole } from '../data/profile';
import { useBackdropDismiss } from '../backdrop';
import '../Profile.css';

/** Profile settings as a pop-up modal, opened from the sidebar name. */
export function ProfileModal({ onClose }: { onClose: () => void }) {
  const { user, profile } = useAuth();
  const uid = user!.uid;
  const [name, setName] = useState(profile?.displayName ?? '');
  const [role, setRoleState] = useState<'player' | 'teacher'>(profile?.role === 'teacher' ? 'teacher' : 'player');
  const [saving, setSaving] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev; };
  }, [onClose]);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await setDisplayName(uid, name.trim());
      await setRole(uid, role);
      setFlash('Saved. Reload to update navigation.');
    } finally {
      setSaving(false);
    }
  };

  return createPortal(
    <div className="profile-modal-backdrop" {...useBackdropDismiss(onClose)} role="presentation">
      <div
        className="profile-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Profile"
      >
        <button type="button" className="profile-modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>Profile</h2>
        <form className="profile-form" onSubmit={save}>
          <label className="profile-row">
            <span className="profile-label">Display name</span>
            <input
              className="profile-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name (shown to teachers on submissions)"
              maxLength={64}
            />
          </label>

          <label className="profile-row">
            <span className="profile-label">Default view</span>
            <select className="profile-input" value={role}
              onChange={(e) => setRoleState(e.target.value as 'player' | 'teacher')}>
              <option value="player">Player</option>
              <option value="teacher">Teacher</option>
            </select>
          </label>

          <p className="profile-email">Signed in as {user?.email}</p>

          <div className="profile-actions">
            <button type="submit" className="profile-save" disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </button>
            {flash && <span className="profile-flash">{flash}</span>}
          </div>
        </form>
      </div>
    </div>,
    document.body,
  );
}
