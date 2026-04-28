import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Board } from './Board';
import {
  api,
  type AttemptWithProblem,
  type Collection,
  type IngestJob,
  type Submission,
  type Teacher,
} from './api';
import { computeNumberedOverlay } from './numberedMoves';
import type { Stone } from './types';
import './Home.css';

function PlusIcon() {
  return (
    <svg
      width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke="currentColor"
      strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true"
    >
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function formatDate(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export function Home() {
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [teachers, setTeachers] = useState<Teacher[] | null>(null);
  const [batchItems, setBatchItems] = useState<AttemptWithProblem[] | null>(null);
  const [submissions, setSubmissions] = useState<Submission[] | null>(null);
  const [pickedTeacher, setPickedTeacher] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [submitFlash, setSubmitFlash] = useState<string | null>(null);
  const [revealedToken, setRevealedToken] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [newLabel, setNewLabel] = useState('');
  const [creating, setCreating] = useState(false);
  const [showAddTeacher, setShowAddTeacher] = useState(false);
  const [outboxExpanded, setOutboxExpanded] = useState(false);
  const [jobs, setJobs] = useState<IngestJob[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [pollNonce, setPollNonce] = useState(0);
  const prevJobsRef = useRef<IngestJob[] | null>(null);

  useEffect(() => {
    api.tsumego.listCollections()
      .then(setCollections)
      .catch((e) => { setError(String(e)); setCollections([]); });
    api.study.listTeachers()
      .then(setTeachers)
      .catch(() => setTeachers([]));
    api.study.getBatch()
      .then(setBatchItems)
      .catch(() => setBatchItems([]));
    api.study.listSubmissions()
      .then(setSubmissions)
      .catch(() => setSubmissions([]));
  }, []);

  // Poll active ingest jobs. The chain stops automatically once every
  // job is in a terminal phase; bumping `pollNonce` (e.g. after Restart)
  // restarts it.
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      if (cancelled) return;
      try {
        const next = await api.pdf.listJobs();
        if (cancelled) return;
        setJobs(next);
        const hasInFlight = next.some(
          (j) => j.phase === 'rendering' || j.phase === 'detecting',
        );
        if (hasInFlight) timer = window.setTimeout(poll, 2000);
      } catch {
        if (!cancelled) timer = window.setTimeout(poll, 5000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [pollNonce]);

  // When a job transitions to done, refresh the collection list so the
  // newly imported source shows up in the Problems section.
  useEffect(() => {
    const prev = prevJobsRef.current;
    prevJobsRef.current = jobs;
    if (!prev || !jobs) return;
    const becameDone = jobs.some((j) => {
      if (j.phase !== 'done') return false;
      const prevJ = prev.find((p) => p.job_id === j.job_id);
      return !!prevJ && prevJ.phase !== 'done';
    });
    if (becameDone) {
      api.tsumego.listCollections().then(setCollections).catch(() => {});
    }
  }, [jobs]);

  const restartJob = async (job: IngestJob) => {
    setJobsError(null);
    try {
      await api.pdf.restartJob(job.job_id);
      setPollNonce((n) => n + 1);
    } catch (e) {
      setJobsError(`Couldn't restart: ${e}`);
    }
  };

  const dismissJob = async (job: IngestJob) => {
    setJobsError(null);
    try {
      await api.pdf.dismissJob(job.job_id);
      setJobs((prev) => (prev ?? []).filter((j) => j.job_id !== job.job_id));
    } catch (e) {
      setJobsError(`Couldn't dismiss: ${e}`);
    }
  };

  // Default the radio selection to the first teacher; clear if that
  // teacher gets removed mid-session.
  useEffect(() => {
    if (!teachers) return;
    setPickedTeacher((cur) => {
      if (cur && teachers.some((t) => t.id === cur)) return cur;
      return teachers[0]?.id ?? null;
    });
  }, [teachers]);

  const submitBatch = async () => {
    if (!pickedTeacher || !batchItems || batchItems.length === 0) return;
    setSending(true);
    setError(null);
    setSubmitFlash(null);
    try {
      const r = await api.study.sendBatch(pickedTeacher);
      const label = (teachers ?? []).find((t) => t.id === pickedTeacher)?.label ?? 'teacher';
      setSubmitFlash(`Submitted ${r.sent_count} problem${r.sent_count === 1 ? '' : 's'} to ${label}.`);
      const [items, subs] = await Promise.all([
        api.study.getBatch(),
        api.study.listSubmissions(),
      ]);
      setBatchItems(items);
      setSubmissions(subs);
    } catch (e) {
      setError(`Submit failed: ${e}`);
    } finally {
      setSending(false);
    }
  };

  const addTeacher = async () => {
    const label = newLabel.trim();
    if (!label) return;
    setCreating(true);
    setError(null);
    try {
      const t = await api.study.createTeacher(label);
      setTeachers((prev) => [...(prev ?? []), t]);
      setNewLabel('');
      setRevealedToken(t.id);
      setShowAddTeacher(false);
    } catch (e) {
      setError(`Couldn't add teacher: ${e}`);
    } finally {
      setCreating(false);
    }
  };

  const removeTeacher = async (t: Teacher) => {
    if (!confirm(`Remove “${t.label}”? Their existing link will stop working.`)) return;
    try {
      await api.study.deleteTeacher(t.id);
      setTeachers((prev) => (prev ?? []).filter((x) => x.id !== t.id));
    } catch (e) {
      setError(`Couldn't remove teacher: ${e}`);
    }
  };

  const copyUrl = async (t: Teacher) => {
    const abs = new URL(t.url, window.location.origin).toString();
    try {
      await navigator.clipboard.writeText(abs);
      setCopiedId(t.id);
      setTimeout(() => setCopiedId((id) => (id === t.id ? null : id)), 2000);
    } catch {
      // clipboard unavailable; URL is already visible for manual copy
    }
  };

  const deleteCollection = async (c: Collection) => {
    const ok = confirm(
      `Delete ${c.count} problem${c.count === 1 ? '' : 's'} from “${c.source}”?`
    );
    if (!ok) return;
    setBusy(c.source);
    try {
      await api.tsumego.deleteCollection(c.source);
      const next = await api.tsumego.listCollections();
      setCollections(next);
    } catch (e) {
      setError(`Delete failed: ${e}`);
    } finally {
      setBusy(null);
    }
  };

  const teacherUrl = (t: Teacher) => new URL(t.url, window.location.origin).toString();

  return (
    <div className="home">
      <header className="home-header">
        <h1>Go problem workbook</h1>
        <nav>
          <Link to="/testing" className="dim">developer tools</Link>
        </nav>
      </header>

      <section className="home-section teachers-section">
        <div className="section-heading">
          <h2>Teachers</h2>
          <button
            type="button"
            className="section-add-btn"
            onClick={() => setShowAddTeacher((v) => !v)}
            aria-label="Add a teacher"
            title="Add a teacher"
            aria-expanded={showAddTeacher}
          >
            <PlusIcon />
          </button>
        </div>
        <div className="section-body">
        {teachers === null && <p className="dim">Loading…</p>}
        {teachers !== null && teachers.length === 0 && !showAddTeacher && (
          <p className="dim">
            No teachers yet. Click + to add one.
          </p>
        )}
        {teachers !== null && teachers.length > 0 && (
          <ul className="teachers-list">
            {teachers.map((t) => (
              <li key={t.id} className="teacher-row">
                <div className="teacher-row-main">
                  <span className="teacher-label">{t.label}</span>
                  <span className="teacher-added">added {formatDate(t.created_at)}</span>
                </div>
                <div className="teacher-row-actions">
                  <button
                    type="button"
                    onClick={() => setRevealedToken((cur) => cur === t.id ? null : t.id)}
                    className="teacher-link-btn"
                  >
                    {revealedToken === t.id ? 'Hide link' : 'Show link'}
                  </button>
                  <button
                    type="button"
                    onClick={() => removeTeacher(t)}
                    className="teacher-remove-btn"
                    aria-label={`Remove ${t.label}`}
                    title="Remove teacher"
                  >
                    ×
                  </button>
                </div>
                {revealedToken === t.id && (
                  <div className="teacher-url-row">
                    <code className="teacher-url">{teacherUrl(t)}</code>
                    <button type="button" onClick={() => copyUrl(t)} className="teacher-url-copy">
                      {copiedId === t.id ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
        {showAddTeacher && (
          <form
            className="add-teacher"
            onSubmit={(e) => { e.preventDefault(); addTeacher(); }}
          >
            <input
              type="text"
              placeholder="Name"
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              disabled={creating}
              autoFocus
            />
            <button type="submit" disabled={creating || !newLabel.trim()}>
              {creating ? 'Adding…' : 'Add'}
            </button>
            <button
              type="button"
              className="add-teacher-cancel"
              onClick={() => { setShowAddTeacher(false); setNewLabel(''); }}
              disabled={creating}
            >
              Cancel
            </button>
          </form>
        )}
        <p className="teachers-note">
          Each teacher gets a unique link. Anyone with the link can grade
          your submissions, so treat it like a password.
        </p>
        </div>
      </section>

      <section className="home-section collections">
        <div className="section-heading">
          <h2>Problems</h2>
          <Link
            to="/upload"
            className="section-add-btn"
            aria-label="Add problems"
            title="Add problems"
          >
            <PlusIcon />
          </Link>
        </div>
        <div className="section-body">
        {collections === null && <p className="dim">Loading…</p>}
        {collections !== null && collections.length === 0 && (
          <p className="dim">
            No problems yet. Click + to import some.
          </p>
        )}
        {collections !== null && collections.length > 0 && (
          <ul className="collection-list">
            {collections.map((c) => (
              <li key={c.source} className="collection-item">
                <Link
                  to={`/collections/${encodeURIComponent(c.source)}/solve`}
                  className="collection-link"
                >
                  <div className="collection-info">
                    <div className="collection-source">{c.source}</div>
                    <div className="collection-meta">
                      {c.count} {c.count === 1 ? 'problem' : 'problems'}
                      {c.last_uploaded_at && (
                        <>
                          <span className="dot">·</span>
                          <span>added {formatDate(c.last_uploaded_at)}</span>
                        </>
                      )}
                    </div>
                  </div>
                </Link>
                <Link
                  to={`/collections/${encodeURIComponent(c.source)}`}
                  className="collection-edit"
                  aria-label={`Edit collection ${c.source}`}
                  title="Open the QA / edit grid"
                >
                  <svg
                    width="16" height="16" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  >
                    <path d="M12 20h9" />
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
                  </svg>
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
        {jobs && jobs.length > 0 && (
          <ul className="jobs-list">
            {jobs.map((j) => (
              <JobCard
                key={j.job_id}
                job={j}
                onRestart={() => restartJob(j)}
                onDismiss={() => dismissJob(j)}
              />
            ))}
          </ul>
        )}
        {jobsError && <p className="error">{jobsError}</p>}
        {error && <p className="error">{error}</p>}
        </div>
      </section>

      <section className="home-section submissions-section">
        <div className="section-heading">
          <h2>Submissions</h2>
        </div>
        <div className="section-body">

        {batchItems && batchItems.length > 0 && (
          <div className="submissions-outbox">
            <button
              type="button"
              className="submissions-outbox-summary"
              onClick={() => setOutboxExpanded((v) => !v)}
              aria-expanded={outboxExpanded}
              aria-controls="submissions-outbox-grid"
            >
              <span className="submissions-outbox-title">
                To submit
                <span className="submissions-outbox-count">
                  {batchItems.length} problem{batchItems.length === 1 ? '' : 's'}
                </span>
              </span>
              <span className="submissions-outbox-chevron" aria-hidden="true">
                {outboxExpanded ? '▾' : '▸'}
              </span>
            </button>
            {teachers && teachers.length > 0 ? (
              <div className="submissions-outbox-actions">
                <span className="submissions-outbox-label">Send to:</span>
                <div className="submissions-outbox-pick">
                  {teachers.map((t) => (
                    <label key={t.id} className="submissions-outbox-radio">
                      <input
                        type="radio"
                        name="submissions-teacher"
                        checked={pickedTeacher === t.id}
                        onChange={() => setPickedTeacher(t.id)}
                        disabled={sending}
                      />
                      <span>{t.label}</span>
                    </label>
                  ))}
                </div>
                <button
                  type="button"
                  className="submissions-submit-btn"
                  onClick={submitBatch}
                  disabled={sending || !pickedTeacher}
                >
                  {sending ? 'Submitting…' : 'Submit'}
                </button>
              </div>
            ) : (
              <p className="submissions-outbox-warn">
                Add a teacher above before submitting.
              </p>
            )}
            {submitFlash && <p className="submissions-flash">{submitFlash}</p>}
            {outboxExpanded && (
              <ul id="submissions-outbox-grid" className="submissions-outbox-grid">
                {batchItems.map((it) => (
                  <SubmissionPreviewTile key={it.attempt.id} item={it} />
                ))}
              </ul>
            )}
          </div>
        )}

        {submissions && submissions.length > 0 && (
          <ul className="submissions-list">
            {submissions.map((s) => (
              <SubmissionRow
                key={s.sent_at}
                submission={s}
                teachers={teachers ?? []}
              />
            ))}
          </ul>
        )}

        {(!batchItems || batchItems.length === 0)
          && (!submissions || submissions.length === 0) && (
          <p className="dim">
            Save a problem for submission from the solver to see it here.
          </p>
        )}

        <div className="submissions-footer">
          <Link to="/reviewed" className="submissions-history-link">
            Submission history →
          </Link>
        </div>
        </div>
      </section>
    </div>
  );
}

function SubmissionPreviewTile({ item }: { item: AttemptWithProblem }) {
  const stones: Stone[] = (item.problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const moves = item.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(moves);
  const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
  const viewport = boundingViewport(allPts);
  return (
    <li className="submissions-outbox-tile">
      <div className="submissions-outbox-tile-board">
        <Board
          stones={stones}
          numberedMoves={overlay.boardNumbers}
          viewport={viewport}
          displayOnly
        />
      </div>
      <div className="submissions-outbox-tile-meta">
        <span className="submissions-outbox-tile-source">{item.problem.source}</span>
        <span className="submissions-outbox-tile-idx">
          #{item.problem.source_board_idx + 1}
          <span className="dot">·</span>
          {moves.length} move{moves.length === 1 ? '' : 's'}
        </span>
      </div>
    </li>
  );
}

function boundingViewport(points: { x: number; y: number }[]) {
  if (points.length === 0) {
    return { colMin: 0, colMax: 18, rowMin: 0, rowMax: 18 };
  }
  let xmin = 18, xmax = 0, ymin = 18, ymax = 0;
  for (const p of points) {
    if (p.x < xmin) xmin = p.x;
    if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.y > ymax) ymax = p.y;
  }
  return {
    colMin: Math.max(0, xmin - 1),
    colMax: Math.min(18, xmax + 1),
    rowMin: Math.max(0, ymin - 1),
    rowMax: Math.min(18, ymax + 1),
  };
}

function JobCard({
  job, onRestart, onDismiss,
}: {
  job: IngestJob;
  onRestart: () => void;
  onDismiss: () => void;
}) {
  const isError = job.phase === 'error';
  const isDone = job.phase === 'done';
  const isStalled = job.stalled && !isError && !isDone;
  const showRestart = isError || isStalled;
  const showDismiss = isError || isDone || isStalled;

  let label: string;
  let frac: number | null = null;
  let indeterminate = false;
  if (isError) {
    label = `Failed: ${job.error ?? 'unknown error'}`;
  } else if (isStalled) {
    label = 'Looks stalled — try restarting.';
  } else if (job.phase === 'rendering') {
    if (job.total_pages) {
      label = `Rendering pages · ${job.pages_rendered} / ${job.total_pages}`;
      frac = job.pages_rendered / job.total_pages;
    } else {
      label = 'Rendering pages…';
      indeterminate = true;
    }
  } else if (job.phase === 'detecting') {
    const saved = job.total_saved;
    if (job.total_pages) {
      label = `Detecting boards · page ${job.pages_detected} / ${job.total_pages}`
        + ` · ${saved} saved`;
      frac = job.pages_detected / job.total_pages;
    } else {
      label = `Detecting boards · ${saved} saved`;
      indeterminate = true;
    }
  } else {
    // done
    const saved = job.total_saved;
    const skipped = job.skipped;
    label = `${saved} problem${saved === 1 ? '' : 's'} imported`
      + (skipped > 0 ? ` · ${skipped} already present` : '');
    frac = 1;
  }

  const stateClass = isError
    ? 'state-error'
    : isDone
      ? 'state-done'
      : isStalled ? 'state-stalled' : 'state-active';

  return (
    <li className={`job-card ${stateClass}`}>
      <div className="job-card-main">
        <div className="job-card-source">{job.source}</div>
        <div className="job-card-status">{label}</div>
        {!isError && (
          indeterminate
            ? <progress className="job-progress" />
            : <progress className="job-progress" value={frac ?? 0} max={1} />
        )}
      </div>
      <div className="job-card-actions">
        {isDone && (
          <Link
            to={`/collections/${encodeURIComponent(job.source)}`}
            className="job-link"
          >
            Open
          </Link>
        )}
        {showRestart && (
          <button type="button" className="job-restart" onClick={onRestart}>
            Restart
          </button>
        )}
        {showDismiss && (
          <button
            type="button"
            className="job-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss"
            title="Dismiss"
          >
            ×
          </button>
        )}
      </div>
    </li>
  );
}

function SubmissionRow({
  submission, teachers,
}: {
  submission: Submission;
  teachers: Teacher[];
}) {
  const teacher = teachers.find((t) => t.id === submission.teacher_id);
  const teacherLabel = teacher?.label ?? '(removed teacher)';
  const total = submission.items.length;
  const reviewed = submission.items.filter(
    (it) => it.attempt.reviews[submission.teacher_id] !== undefined,
  ).length;
  const stateLabel = submission.state === 'returned' ? 'Ready to view' : 'Pending review';
  const stateClass = `submissions-state state-${submission.state}`;
  return (
    <li className="submissions-row">
      <Link
        to={`/submissions/${encodeURIComponent(submission.sent_at)}`}
        className="submissions-row-link"
      >
        <div className="submissions-row-main">
          <span className={stateClass}>{stateLabel}</span>
          <span className="submissions-row-teacher">{teacherLabel}</span>
          <span className="submissions-row-when">
            submitted {formatDate(submission.sent_at)}
          </span>
        </div>
        <div className="submissions-row-meta">
          {submission.state === 'pending'
            ? `${reviewed} of ${total} reviewed`
            : `${total} problem${total === 1 ? '' : 's'}`}
        </div>
      </Link>
    </li>
  );
}
