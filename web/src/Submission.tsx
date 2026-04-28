import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Board } from './Board';
import {
  api,
  type AttemptWithProblem,
  type ProblemStatus,
  type Submission as SubmissionT,
} from './api';
import { computeNumberedOverlay } from './numberedMoves';
import type { Stone } from './types';
import './Reviewed.css';

/** Detail view for a single submission. Shows every attempt in the
 * submission with its verdict (if reviewed). When all attempts are
 * reviewed, the student can mark the submission as read; that moves
 * the submission out of the in-flight panel and into graded history. */
export function Submission() {
  const { sent_at: encSentAt = '' } = useParams();
  const sentAt = decodeURIComponent(encSentAt);
  const navigate = useNavigate();
  const [submission, setSubmission] = useState<SubmissionT | null>(null);
  const [statuses, setStatuses] = useState<Record<string, ProblemStatus>>({});
  const [error, setError] = useState<string | null>(null);
  const [acking, setAcking] = useState(false);

  useEffect(() => {
    api.study.getSubmission(sentAt)
      .then(setSubmission)
      .catch((e) => setError(String(e)));
    api.study.problemStatuses()
      .then(setStatuses)
      .catch(() => {});
  }, [sentAt]);

  const ack = async () => {
    if (!submission) return;
    setAcking(true);
    try {
      await api.study.ackSubmission(submission.sent_at);
      navigate('/');
    } catch (e) {
      setError(String(e));
      setAcking(false);
    }
  };

  if (submission === null) {
    return (
      <div className="reviewed">
        <p className="dim">{error ?? 'Loading…'}</p>
      </div>
    );
  }

  const teacherLabel = submission.reviewer_name;
  const total = submission.items.length;
  const reviewed = submission.items.filter(
    (it) => it.attempt.reviews[submission.reviewer_id] !== undefined,
  ).length;
  const incorrect = submission.items.filter(
    (it) => it.attempt.reviews[submission.reviewer_id]?.verdict === 'incorrect',
  );
  const firstIncorrect = incorrect[0];
  const retryHref = firstIncorrect
    ? `/collections/${encodeURIComponent(firstIncorrect.problem.source)}/solve/${firstIncorrect.problem.id}?from_submission=${encodeURIComponent(submission.sent_at)}&retry=incorrect`
    : null;

  return (
    <div className="reviewed">
      <header className="reviewed-header">
        <Link to="/" className="back-link">← home</Link>
        <h1>Submission</h1>
        <div className="reviewed-meta">
          Sent to <strong>{teacherLabel}</strong> on {formatTimestamp(submission.sent_at)}
          {' · '}
          {submission.state === 'pending'
            ? `${reviewed} of ${total} reviewed`
            : `${total} problem${total === 1 ? '' : 's'}, all reviewed`}
        </div>
        {submission.state === 'returned' && (
          <div className="submission-actions">
            {retryHref && (
              <Link to={retryHref} className="submission-retry-btn">
                Retry incorrect ({incorrect.length})
              </Link>
            )}
            <button
              type="button"
              onClick={ack}
              disabled={acking}
              className="submission-ack-btn"
            >
              {acking ? 'Marking…' : 'Mark as read'}
            </button>
          </div>
        )}
      </header>

      {error && <div className="reviewed-error">{error}</div>}

      <ul className="reviewed-list">
        {submission.items.map((it) => (
          <SubmissionRow
            key={it.attempt.id}
            item={it}
            teacherId={submission.reviewer_id}
            teacherLabel={teacherLabel}
            status={statuses[it.problem.id] ?? null}
          />
        ))}
      </ul>
    </div>
  );
}

function SubmissionRow({
  item, teacherId, teacherLabel, status,
}: {
  item: AttemptWithProblem;
  teacherId: string;
  teacherLabel: string;
  status: ProblemStatus | null;
}) {
  const stones: Stone[] = (item.problem.stones ?? []).map((s) => ({
    x: s.col, y: s.row, color: s.color as 'B' | 'W',
  }));
  const moves = item.attempt.moves.map((m) => ({ x: m.col, y: m.row }));
  const overlay = computeNumberedOverlay(moves);
  const allPts = [...stones.map((s) => ({ x: s.x, y: s.y })), ...moves];
  const viewport = boundingViewport(allPts);
  const review = item.attempt.reviews[teacherId];
  const retried = review?.verdict === 'incorrect'
    && status?.latest_attempt_at != null
    && status.latest_attempt_at > item.attempt.submitted_at;
  const solveHref = `/collections/${encodeURIComponent(item.problem.source)}/solve/${item.problem.id}?from_submission=${encodeURIComponent(item.attempt.sent_at ?? '')}`;

  return (
    <li className="reviewed-row">
      <Link to={solveHref} className="reviewed-row-link" aria-label="Open in solver">
        <div className="reviewed-row-board">
          <Board
            stones={stones}
            numberedMoves={overlay.boardNumbers}
            viewport={viewport}
            displayOnly
          />
        </div>
        <div className="reviewed-row-info">
          <div className="reviewed-row-source">
            <span className="reviewed-row-collection">{item.problem.source}</span>
            <span className="dot">·</span>
            <span>Problem {item.problem.source_board_idx + 1}</span>
          </div>
          <div className="reviewed-row-stamp">
            {moves.length} move{moves.length === 1 ? '' : 's'}
            <span className="dot">·</span>
            submitted {formatTimestamp(item.attempt.submitted_at)}
          </div>
          <ul className="reviewed-verdicts">
            {review ? (
              <li className={`reviewed-verdict v-${review.verdict}`}>
                <span className="reviewed-verdict-mark">
                  {review.verdict === 'correct' ? '✓' : '✗'}
                </span>
                <span className="reviewed-verdict-label">{teacherLabel}</span>
                <span className="reviewed-verdict-when">
                  {formatTimestamp(review.reviewed_at)}
                </span>
              </li>
            ) : (
              <li className="reviewed-verdict v-pending">
                <span className="reviewed-verdict-mark">…</span>
                <span className="reviewed-verdict-label">awaiting {teacherLabel}</span>
              </li>
            )}
            {retried && (
              <li className="reviewed-verdict v-retried">
                <span className="reviewed-verdict-mark">↻</span>
                <span className="reviewed-verdict-label">incorrect but retried</span>
              </li>
            )}
          </ul>
        </div>
      </Link>
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

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
