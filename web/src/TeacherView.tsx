import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Board } from './Board';
import { api, type LinkedUser, type TeacherAttemptWithProblem } from './api';
import { computeNumberedOverlay } from './numberedMoves';
import type { Stone } from './types';
import './Home.css';
import './TeacherView.css';

type Verdict = 'correct' | 'incorrect';

/** Authenticated reviewer view.
 *
 * Without `:student_uid`: a landing page that mirrors the student-side
 * Home layout, with Students and Submissions sections.
 *
 * With `:student_uid`: a per-student review flow with three modes —
 *   1. entry  — list of pending submissions (one per `sent_at`)
 *   2. grid   — tile grid for one submission, with verdict buttons
 *   3. detail — full board for one attempt, with prev/next inside submission
 */
export function TeacherView() {
  const { student_uid: encStudentUid } = useParams();
  const studentUid = encStudentUid ? decodeURIComponent(encStudentUid) : null;

  const [students, setStudents] = useState<LinkedUser[] | null>(null);
  const [studentsError, setStudentsError] = useState<string | null>(null);

  useEffect(() => {
    api.teacher.listStudents()
      .then(setStudents)
      .catch((e) => setStudentsError(String(e)));
  }, []);

  if (studentsError) {
    return (
      <div className="teacher">
        <p className="teacher-error">{studentsError}</p>
      </div>
    );
  }

  if (students === null) {
    return <div className="teacher" style={{ color: '#666' }}>Loading…</div>;
  }

  if (!studentUid) {
    return <TeacherLanding students={students} />;
  }

  const student = students.find((s) => s.user_id === studentUid) ?? null;
  if (!student) {
    return (
      <div className="teacher">
        <p className="teacher-error">Not linked as teacher of this student.</p>
        <Link to="/teacher" className="back-link">← teacher view</Link>
      </div>
    );
  }

  return <StudentReview student={student} />;
}

type PendingBatch = {
  student: LinkedUser;
  sent_at: string;
  count: number;
};

function TeacherLanding({ students }: { students: LinkedUser[] }) {
  const [pending, setPending] = useState<PendingBatch[] | null>(null);
  const [myName, setMyName] = useState<string | null>(null);

  useEffect(() => {
    api.study.getProfile()
      .then((p) => setMyName(p.display_name ?? ''))
      .catch(() => setMyName(''));
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const all = await Promise.all(
        students.map(async (s) => {
          try {
            const items = await api.teacher.queue(s.user_id);
            return { student: s, items };
          } catch {
            return { student: s, items: [] as TeacherAttemptWithProblem[] };
          }
        }),
      );
      if (cancelled) return;
      const flat: PendingBatch[] = [];
      for (const { student, items } of all) {
        for (const b of groupBatches(items)) {
          flat.push({ student, sent_at: b.sent_at, count: b.items.length });
        }
      }
      flat.sort((a, b) => (b.sent_at || '').localeCompare(a.sent_at || ''));
      setPending(flat);
    })();
    return () => { cancelled = true; };
  }, [students]);

  return (
    <div className="home">
      <header className="home-header">
        <h1>{myName || '\u00a0'}</h1>
        <nav className="home-nav">
          <Link to="/" state={{ from: 'teacher' }} className="dim">
            student view
          </Link>
          <Link to="/profile" className="dim">profile</Link>
          <Link to="/testing" className="dim">developer tools</Link>
        </nav>
      </header>

      <section className="home-section teachers-section">
        <div className="section-heading">
          <h2>Students</h2>
        </div>
        <div className="section-body">
          {students.length === 0 ? (
            <p className="dim">
              No students have linked you yet.
            </p>
          ) : (
            <ul className="teachers-list">
              {students.map((s) => (
                <li key={s.user_id} className="teacher-row">
                  <Link
                    to={`/teacher/students/${encodeURIComponent(s.user_id)}`}
                    className="teacher-row-link"
                  >
                    <span className="teacher-label">{s.display_name}</span>
                    {s.email && (
                      <span className="teacher-email">{s.email}</span>
                    )}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="home-section submissions-section">
        <div className="section-heading">
          <h2>Submissions</h2>
        </div>
        <div className="section-body">
          {pending === null && <p className="dim">Loading…</p>}
          {pending !== null && pending.length === 0 && (
            <p className="dim">All caught up — nothing pending right now.</p>
          )}
          {pending !== null && pending.length > 0 && (
            <ul className="submissions-list">
              {pending.map((p) => (
                <li
                  key={`${p.student.user_id}|${p.sent_at}`}
                  className="submissions-row"
                >
                  <Link
                    to={`/teacher/students/${encodeURIComponent(p.student.user_id)}`}
                    className="submissions-row-link"
                  >
                    <div className="submissions-row-main">
                      <span className="submissions-state state-pending">
                        Pending review
                      </span>
                      <span className="submissions-row-teacher">
                        {p.student.display_name}
                      </span>
                      <span className="submissions-row-when">
                        submitted {formatTimestamp(p.sent_at)}
                      </span>
                    </div>
                    <div className="submissions-row-meta">
                      {p.count} problem{p.count === 1 ? '' : 's'}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}

function StudentReview({ student }: { student: LinkedUser }) {
  const studentUid = student.user_id;
  const [items, setItems] = useState<TeacherAttemptWithProblem[] | null>(null);
  const [history, setHistory] = useState<TeacherAttemptWithProblem[] | null>(null);
  const [drafts, setDrafts] = useState<Map<string, Verdict>>(new Map());
  const [selectedBatch, setSelectedBatch] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    const [q, h] = await Promise.all([
      api.teacher.queue(studentUid),
      api.teacher.reviewed(studentUid),
    ]);
    setItems(q);
    setHistory(h);
  };

  useEffect(() => {
    setItems(null);
    setHistory(null);
    refresh().catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studentUid]);

  const batches = useMemo(() => groupBatches(items ?? []), [items]);
  const currentBatch = selectedBatch
    ? batches.find((b) => b.sent_at === selectedBatch) ?? null
    : null;

  const setVerdict = (attemptId: string, v: Verdict | null) => {
    setDrafts((prev) => {
      const next = new Map(prev);
      if (v === null) next.delete(attemptId);
      else next.set(attemptId, v);
      return next;
    });
  };

  const submitBatch = async () => {
    if (!currentBatch) return;
    const toSend: Array<[string, Verdict]> = [];
    for (const it of currentBatch.items) {
      const v = drafts.get(it.attempt.id);
      if (v) toSend.push([it.attempt.id, v]);
    }
    if (toSend.length === 0) return;
    setSubmitting(true);
    setError(null);
    setFlash(null);
    try {
      for (const [aid, verdict] of toSend) {
        await api.teacher.review(studentUid, aid, verdict);
      }
      setDrafts((prev) => {
        const next = new Map(prev);
        for (const [aid] of toSend) next.delete(aid);
        return next;
      });
      setFlash(`Submitted ${toSend.length} review${toSend.length === 1 ? '' : 's'}.`);
      setSelectedIdx(null);
      setSelectedBatch(null);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };


  const backLink = (
    <Link to="/teacher" className="back-link">← teacher view</Link>
  );

  if (error && !items) {
    return (
      <div className="teacher">
        {backLink}
        <p className="teacher-error">{error}</p>
      </div>
    );
  }
  if (items === null) {
    return <div className="teacher" style={{ color: '#666' }}>Loading…</div>;
  }

  if (showHistory) {
    return (
      <HistoryView
        student={student}
        items={history ?? []}
        onBack={() => setShowHistory(false)}
      />
    );
  }

  if (items.length === 0) {
    return (
      <div className="teacher">
        <header className="teacher-header">
          <div>
            {backLink}
            <h1>Review submissions</h1>
            <div className="teacher-meta">
              For <strong>{student.display_name}</strong>
            </div>
          </div>
        </header>
        {flash && <div className="teacher-flash">{flash}</div>}
        <p className="teacher-empty">All caught up — nothing pending right now.</p>
        {(history?.length ?? 0) > 0 && (
          <button
            type="button"
            className="teacher-history-link"
            onClick={() => setShowHistory(true)}
          >
            View submission history ({history!.length})
          </button>
        )}
      </div>
    );
  }

  if (currentBatch && selectedIdx !== null) {
    return (
      <DetailView
        student={student}
        batch={currentBatch}
        idx={selectedIdx}
        setIdx={setSelectedIdx}
        drafts={drafts}
        setVerdict={setVerdict}
      />
    );
  }

  if (currentBatch) {
    return (
      <GridView
        student={student}
        batch={currentBatch}
        onBack={() => setSelectedBatch(null)}
        onOpen={(i) => setSelectedIdx(i)}
        drafts={drafts}
        setVerdict={setVerdict}
        submitBatch={submitBatch}
        submitting={submitting}
        flash={flash}
        error={error}
      />
    );
  }

  // Entry: batch list.
  const totalDrafts = drafts.size;
  return (
    <div className="teacher">
      <header className="teacher-header">
        <div>
          {backLink}
          <h1>Review submissions</h1>
          <div className="teacher-meta">
            For <strong>{student.display_name}</strong> &nbsp;·&nbsp;
            {batches.length} submission{batches.length === 1 ? '' : 's'} pending
          </div>
        </div>
      </header>

      {totalDrafts > 0 && (
        <div className="teacher-draft-banner">
          {totalDrafts} draft{totalDrafts === 1 ? '' : 's'} across submissions —
          open a submission to grade.
        </div>
      )}
      {flash && <div className="teacher-flash">{flash}</div>}
      {error && <div className="teacher-error">{error}</div>}

      {(history?.length ?? 0) > 0 && (
        <button
          type="button"
          className="teacher-history-link"
          onClick={() => setShowHistory(true)}
        >
          View graded history ({history!.length})
        </button>
      )}

      <ul className="teacher-batch-list">
        {batches.map((b) => {
          const draftCount = b.items.filter((it) => drafts.has(it.attempt.id)).length;
          return (
            <li key={b.sent_at} className="teacher-batch-row">
              <button
                type="button"
                className="teacher-batch-btn"
                onClick={() => { setSelectedBatch(b.sent_at); setSelectedIdx(null); }}
              >
                <div className="teacher-batch-when">
                  New submission from {student.display_name}
                </div>
                {b.sent_at && (
                  <div className="teacher-batch-sent">
                    sent {formatTimestamp(b.sent_at)}
                  </div>
                )}
                <div className="teacher-batch-counts">
                  {b.items.length} problem{b.items.length === 1 ? '' : 's'}
                  {draftCount > 0 && (
                    <>
                      <span className="dot">·</span>
                      <span className="teacher-batch-drafts">
                        {draftCount} drafted
                      </span>
                    </>
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function groupBatches(items: TeacherAttemptWithProblem[]): {
  sent_at: string;
  items: TeacherAttemptWithProblem[];
}[] {
  const by: Map<string, TeacherAttemptWithProblem[]> = new Map();
  for (const it of items) {
    const k = it.attempt.sent_at ?? '';
    if (!by.has(k)) by.set(k, []);
    by.get(k)!.push(it);
  }
  const out = Array.from(by.entries()).map(([sent_at, items]) => ({
    sent_at,
    items,
  }));
  out.sort((a, b) => (b.sent_at || '').localeCompare(a.sent_at || ''));
  return out;
}

function GridView({
  student, batch, onBack, onOpen, drafts, setVerdict, submitBatch, submitting,
  flash, error,
}: {
  student: LinkedUser;
  batch: { sent_at: string; items: TeacherAttemptWithProblem[] };
  onBack: () => void;
  onOpen: (i: number) => void;
  drafts: Map<string, Verdict>;
  setVerdict: (aid: string, v: Verdict | null) => void;
  submitBatch: () => void;
  submitting: boolean;
  flash: string | null;
  error: string | null;
}) {
  const draftCount = batch.items.filter((it) => drafts.has(it.attempt.id)).length;

  return (
    <div className="teacher">
      <header className="teacher-header">
        <div>
          <button type="button" className="back-link" onClick={onBack}>
            ← all submissions
          </button>
          <h1>{batch.sent_at ? `Submission — ${formatTimestamp(batch.sent_at)}` : 'Submission'}</h1>
          <div className="teacher-meta">
            For <strong>{student.display_name}</strong> &nbsp;·&nbsp;
            {batch.items.length} pending
          </div>
        </div>
      </header>

      <SubmitBar
        draftCount={draftCount}
        total={batch.items.length}
        onSubmit={submitBatch}
        submitting={submitting}
      />
      {flash && <div className="teacher-flash">{flash}</div>}
      {error && <div className="teacher-error">{error}</div>}

      <ul className="teacher-grid">
        {batch.items.map((it, i) => (
          <GridTile
            key={it.attempt.id}
            item={it}
            verdict={drafts.get(it.attempt.id) ?? null}
            onOpen={() => onOpen(i)}
            onSetVerdict={(v) => setVerdict(it.attempt.id, v)}
          />
        ))}
      </ul>
    </div>
  );
}

function SubmitBar({
  draftCount, total, onSubmit, submitting,
}: {
  draftCount: number;
  total: number;
  onSubmit: () => void;
  submitting: boolean;
}) {
  return (
    <div className="teacher-submit-bar">
      <span className="teacher-submit-count">
        {draftCount === 0
          ? `${total} pending in this submission.`
          : `${draftCount} of ${total} drafted.`}
      </span>
      <button
        type="button"
        className="teacher-submit-btn"
        onClick={onSubmit}
        disabled={draftCount === 0 || submitting}
      >
        {submitting ? 'Submitting…' : 'Submit reviews'}
      </button>
    </div>
  );
}

function GridTile({
  item, verdict, onOpen, onSetVerdict,
}: {
  item: TeacherAttemptWithProblem;
  verdict: Verdict | null;
  onOpen: () => void;
  onSetVerdict: (v: Verdict | null) => void;
}) {
  const stones: Stone[] = (item.problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const moves = item.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(moves);
  const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
  const viewport = boundingViewport(allPts);

  return (
    <li className={`teacher-tile${verdict ? ` verdict-${verdict}` : ''}`}>
      <button
        type="button"
        className="teacher-tile-board"
        onClick={onOpen}
        aria-label={`Open problem ${item.problem.source_board_idx + 1}`}
      >
        <Board
          stones={stones}
          numberedMoves={overlay.boardNumbers}
          viewport={viewport}
          displayOnly
        />
      </button>
      <div className="teacher-tile-meta">
        {overlay.chains.length > 0 && (
          <ul className="teacher-tile-chains" aria-label="Recaptures">
            {overlay.chains.map((chain, i) => (
              <li key={i} className="teacher-tile-chain">
                {chain.map((n, j) => (
                  <span key={j}>
                    {j > 0 && <span className="teacher-tile-chain-sep">→</span>}
                    <span className="teacher-tile-chain-num">{n}</span>
                  </span>
                ))}
              </li>
            ))}
          </ul>
        )}
        <div className="teacher-tile-source">
          <span className="teacher-tile-collection">{item.problem.source}</span>
          <span className="dot">·</span>
          <span>#{item.problem.source_board_idx + 1}</span>
        </div>
        <div className="teacher-tile-stamp">
          {moves.length} move{moves.length === 1 ? '' : 's'}
        </div>
      </div>
      <div className="teacher-tile-verdicts">
        <button
          type="button"
          className={`mini-verdict correct${verdict === 'correct' ? ' selected' : ''}`}
          onClick={() => onSetVerdict(verdict === 'correct' ? null : 'correct')}
          aria-pressed={verdict === 'correct'}
        >
          ✓
        </button>
        <button
          type="button"
          className={`mini-verdict incorrect${verdict === 'incorrect' ? ' selected' : ''}`}
          onClick={() => onSetVerdict(verdict === 'incorrect' ? null : 'incorrect')}
          aria-pressed={verdict === 'incorrect'}
        >
          ✗
        </button>
      </div>
    </li>
  );
}

function DetailView({
  student, batch, idx, setIdx, drafts, setVerdict,
}: {
  student: LinkedUser;
  batch: { sent_at: string; items: TeacherAttemptWithProblem[] };
  idx: number;
  setIdx: (i: number | null) => void;
  drafts: Map<string, Verdict>;
  setVerdict: (aid: string, v: Verdict | null) => void;
}) {
  const current = batch.items[idx];
  const verdict = drafts.get(current.attempt.id) ?? null;
  const total = batch.items.length;

  const stones: Stone[] = useMemo(() => {
    return (current.problem.stones ?? []).map((s) => ({
      x: s.col, y: s.row, color: s.color as 'B' | 'W',
    }));
  }, [current]);

  const overlay = useMemo(() => {
    const moves = current.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
    return computeNumberedOverlay(moves);
  }, [current]);

  const viewport = useMemo(() => {
    const allPts = [
      ...stones.map((s) => ({ x: s.x, y: s.y })),
      ...current.attempt.moves.map((m) => ({ x: m.col, y: m.row })),
    ];
    return boundingViewport(allPts);
  }, [current, stones]);

  const goPrev = () => { if (idx > 0) setIdx(idx - 1); };
  const goNext = () => { if (idx < total - 1) setIdx(idx + 1); };
  const onPick = (v: Verdict) => {
    setVerdict(current.attempt.id, verdict === v ? null : v);
  };
  // Verdicts are kept as drafts; the teacher submits the whole batch
  // from the grid view. "Save & continue" advances; on the last problem
  // it returns to the grid so the submit bar is visible.
  const saveAndContinue = () => {
    if (!verdict) return;
    if (idx < total - 1) setIdx(idx + 1);
    else setIdx(null);
  };

  return (
    <div className="teacher">
      <header className="teacher-header">
        <div>
          <button type="button" className="back-link" onClick={() => setIdx(null)}>
            ← back to submission
          </button>
          <h1>
            Problem {current.problem.source_board_idx + 1}
            <span className="teacher-counter"> ({idx + 1} / {total})</span>
          </h1>
          <div className="teacher-meta">
            For <strong>{student.display_name}</strong>
          </div>
        </div>
      </header>

      <div className="teacher-problem-meta">
        <span className="teacher-source">{current.problem.source}</span>
        <span className="dot">·</span>
        <span>{current.problem.black_to_play ? 'Black to play' : 'White to play'}</span>
        <span className="dot">·</span>
        <span className="teacher-submitted">
          submitted {formatTimestamp(current.attempt.submitted_at)}
        </span>
      </div>

      <div className="teacher-workspace">
        <div className="teacher-board">
          <Board
            stones={stones}
            numberedMoves={overlay.boardNumbers}
            viewport={viewport}
            displayOnly
          />
        </div>
        {overlay.chains.length > 0 && (
          <aside className="teacher-chains" aria-label="Move sequence">
            <div className="teacher-chains-title">Sequence</div>
            <ol className="teacher-chains-list">
              {overlay.chains.map((chain, i) => (
                <li key={i} className="chain">
                  {chain.map((n, j) => (
                    <span key={j}>
                      {j > 0 && <span className="chain-sep">→</span>}
                      <span className="chain-num">{n}</span>
                    </span>
                  ))}
                </li>
              ))}
            </ol>
          </aside>
        )}
      </div>

      <div className="teacher-actions">
        <button
          className="teacher-step"
          onClick={goPrev}
          disabled={idx === 0}
        >
          ‹ Prev
        </button>
        <button
          className={`verdict correct${verdict === 'correct' ? ' selected' : ''}`}
          onClick={() => onPick('correct')}
        >
          ✓ Correct
        </button>
        <button
          className={`verdict incorrect${verdict === 'incorrect' ? ' selected' : ''}`}
          onClick={() => onPick('incorrect')}
        >
          ✗ Incorrect
        </button>
        <button
          onClick={saveAndContinue}
          disabled={!verdict}
          className="teacher-save-continue"
          aria-label="Save review and continue to next problem"
        >
          Save & continue ›
        </button>
        <button
          className="teacher-step"
          onClick={goNext}
          disabled={idx >= total - 1}
          aria-label="Skip to next problem without saving"
        >
          Skip ›
        </button>
      </div>

      {current.problem.has_image && (
        <details className="teacher-original">
          <summary>View original</summary>
          <img
            src={api.teacher.problemImageUrl(student.user_id, current.problem.id)}
            alt={`Original crop for problem ${current.problem.source_board_idx + 1}`}
          />
        </details>
      )}
    </div>
  );
}

function HistoryView({
  student, items, onBack,
}: {
  student: LinkedUser;
  items: TeacherAttemptWithProblem[];
  onBack: () => void;
}) {
  const [mode, setMode] = useState<'grouped' | 'flat'>('grouped');

  const sorted = useMemo(() => {
    return [...items].sort((a, b) => {
      const ar = a.attempt.review?.reviewed_at ?? '';
      const br = b.attempt.review?.reviewed_at ?? '';
      return br.localeCompare(ar);
    });
  }, [items]);

  const groups = useMemo(() => groupHistoryBySentAt(items), [items]);

  return (
    <div className="teacher">
      <header className="teacher-header">
        <div>
          <button type="button" className="back-link" onClick={onBack}>
            ← back to queue
          </button>
          <h1>Submission history</h1>
          <div className="teacher-meta">
            For <strong>{student.display_name}</strong> &nbsp;·&nbsp;
            {sorted.length} graded
          </div>
          {sorted.length > 0 && (
            <div className="teacher-history-toggle">
              <button
                type="button"
                className={`teacher-history-toggle-btn${mode === 'grouped' ? ' active' : ''}`}
                onClick={() => setMode('grouped')}
              >
                By submission
              </button>
              <button
                type="button"
                className={`teacher-history-toggle-btn${mode === 'flat' ? ' active' : ''}`}
                onClick={() => setMode('flat')}
              >
                View all
              </button>
            </div>
          )}
        </div>
      </header>

      {sorted.length === 0 && (
        <p className="teacher-empty">Nothing graded yet.</p>
      )}

      {sorted.length > 0 && mode === 'flat' && (
        <ul className="teacher-history-list">
          {sorted.map((it) => <HistoryRow key={it.attempt.id} item={it} />)}
        </ul>
      )}

      {sorted.length > 0 && mode === 'grouped' && (
        <div className="teacher-history-groups">
          {groups.map((g) => (
            <section key={g.sent_at || 'unsent'} className="teacher-history-group">
              <h2 className="teacher-history-group-header">
                {g.sent_at
                  ? `Submission — ${formatTimestamp(g.sent_at)}`
                  : 'Submission (unknown date)'}
                <span className="teacher-history-group-count">
                  {g.items.length} problem{g.items.length === 1 ? '' : 's'}
                </span>
              </h2>
              <ul className="teacher-history-list">
                {g.items.map((it) => <HistoryRow key={it.attempt.id} item={it} />)}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryRow({ item }: { item: TeacherAttemptWithProblem }) {
  const stones: Stone[] = (item.problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const moves = item.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(moves);
  const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
  const viewport = boundingViewport(allPts);
  const v = item.attempt.review?.verdict;
  return (
    <li className="teacher-history-row">
      <div className="teacher-history-board">
        <Board
          stones={stones}
          numberedMoves={overlay.boardNumbers}
          viewport={viewport}
          displayOnly
        />
      </div>
      <div className="teacher-history-info">
        <div className="teacher-history-source">
          <span className="teacher-tile-collection">{item.problem.source}</span>
          <span className="dot">·</span>
          <span>Problem {item.problem.source_board_idx + 1}</span>
        </div>
        <div className="teacher-history-stamp">
          {moves.length} move{moves.length === 1 ? '' : 's'}
          {item.attempt.review && (
            <>
              <span className="dot">·</span>
              <span>graded {formatTimestamp(item.attempt.review.reviewed_at)}</span>
            </>
          )}
        </div>
        {v && (
          <div className={`teacher-history-verdict v-${v}`}>
            <span className="teacher-history-mark">
              {v === 'correct' ? '✓' : '✗'}
            </span>
            <span>{v}</span>
          </div>
        )}
      </div>
    </li>
  );
}

function groupHistoryBySentAt(items: TeacherAttemptWithProblem[]): {
  sent_at: string;
  items: TeacherAttemptWithProblem[];
}[] {
  const by = new Map<string, TeacherAttemptWithProblem[]>();
  for (const it of items) {
    const k = it.attempt.sent_at ?? '';
    if (!by.has(k)) by.set(k, []);
    by.get(k)!.push(it);
  }
  for (const arr of by.values()) {
    arr.sort((a, b) => (a.problem.source_board_idx - b.problem.source_board_idx));
  }
  return Array.from(by.entries())
    .map(([sent_at, items]) => ({ sent_at, items }))
    .sort((a, b) => (b.sent_at || '').localeCompare(a.sent_at || ''));
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

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
