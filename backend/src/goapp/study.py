"""Solve attempts, batches, teachers, and per-teacher reviews.

Layout under attempts_dir(user_id):
    attempt_*.json       one attempt per file
    teachers/<id>.json   one teacher (capability URL) per file

Attempt schema:
    {
      "id": "attempt_<ts>_<rand>",
      "problem_id": "tsumego_...",
      "moves": [{"col": int, "row": int}, ...],   ordered, may repeat
      "submitted_at": "<UTC ISO>",
      "sent_to": ["teacher_<id>"],   single recipient (one-element list
                                     for storage compatibility)
      "sent_at": "<UTC ISO>" | null, when the attempt was sent
      "reviews": { "<teacher_id>": {"verdict": "...", "reviewed_at": "..."} },
      "acked_at": "<UTC ISO>" | null,  when the student marked the
                                       reviewed submission as read
    }

Submissions are groups of attempts that share `sent_at`. Each submission
goes to exactly one teacher. State for a submission:
    pending  — at least one attempt has no review from the teacher yet
    returned — all attempts reviewed, but at least one not yet acked
    acked    — student has marked the submission as read

Teacher schema:
    {
      "id": "teacher_<rand>",
      "token": "tt_<urlsafe>",
      "label": "...",
      "created_at": "<UTC ISO>"
    }

Multiple teachers per student. The student's "batch" is just the set of
attempts whose `sent_at` is null — that is, anything submitted since the
last send. Sending stamps `sent_at` and appends teacher ids to `sent_to`
(so re-sending later doesn't duplicate). Each teacher reviews
independently; verdicts are stored per-teacher under `reviews`.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .paths import ATTEMPTS_ROOT, attempts_dir, teacher_path, teachers_dir

VERDICTS = ("correct", "incorrect")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _attempt_path(user_id: str, attempt_id: str) -> Path:
    return attempts_dir(user_id) / f"{attempt_id}.json"


def _normalize_attempt(a: dict) -> dict:
    """Tolerate legacy attempt records (single `review` field, no
    `sent_to`/`reviews`/`sent_at`). Returns a normalized copy without
    rewriting the file — callers that mutate will save normalized."""
    out = dict(a)
    out.setdefault("sent_to", [])
    out.setdefault("sent_at", None)
    out.setdefault("acked_at", None)
    if "reviews" not in out:
        legacy = a.get("review")
        # The pre-multi-teacher review can't be attributed to anyone, so
        # we drop it. (User opted into a clean break.)
        out["reviews"] = {}
        out.pop("review", None)
        _ = legacy
    return out


# --- attempts ---


def save_attempt(user_id: str, problem_id: str, moves: list[dict]) -> dict:
    """Persist a new attempt. Lands unsent (in the student's batch)."""
    udir = attempts_dir(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    aid = f"attempt_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{secrets.token_hex(3)}"
    record = {
        "id": aid,
        "problem_id": problem_id,
        "moves": [{"col": int(m["col"]), "row": int(m["row"])} for m in moves],
        "submitted_at": _now(),
        "sent_to": [],
        "sent_at": None,
        "reviews": {},
        "acked_at": None,
    }
    _attempt_path(user_id, aid).write_text(json.dumps(record, indent=2))
    return record


def load_attempt(user_id: str, attempt_id: str) -> dict | None:
    p = _attempt_path(user_id, attempt_id)
    if not p.exists():
        return None
    try:
        return _normalize_attempt(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return None


def list_attempts(user_id: str) -> list[dict]:
    """Every attempt for the user, oldest first."""
    udir = attempts_dir(user_id)
    if not udir.exists():
        return []
    out: list[dict] = []
    for p in udir.glob("attempt_*.json"):
        try:
            out.append(_normalize_attempt(json.loads(p.read_text())))
        except json.JSONDecodeError:
            continue
    out.sort(key=lambda d: d.get("submitted_at", ""))
    return out


def attempts_for_problem(user_id: str, problem_id: str) -> list[dict]:
    return [a for a in list_attempts(user_id) if a.get("problem_id") == problem_id]


def _latest_by_problem(attempts: list[dict]) -> dict[str, dict]:
    by_pid: dict[str, dict] = {}
    for a in attempts:
        pid = a.get("problem_id")
        if not pid:
            continue
        prev = by_pid.get(pid)
        if prev is None or a.get("submitted_at", "") > prev.get("submitted_at", ""):
            by_pid[pid] = a
    return by_pid


def latest_attempt(user_id: str, problem_id: str) -> dict | None:
    return _latest_by_problem(attempts_for_problem(user_id, problem_id)).get(problem_id)


# --- batch (unsent attempts) ---


def list_unsent(user_id: str) -> list[dict]:
    """Latest attempt per problem with `sent_at is None` — what the
    student would push on the next 'send to teachers' action."""
    unsent = [a for a in list_attempts(user_id) if a.get("sent_at") is None]
    out = list(_latest_by_problem(unsent).values())
    out.sort(key=lambda d: d.get("submitted_at", ""))
    return out


def send_to_teacher(user_id: str, teacher_id: str) -> list[dict]:
    """Send all unsent attempts (latest per problem) to a single teacher
    as one submission. Earlier unsent attempts on the same problem are
    superseded — they get marked sent with no recipient so they don't
    keep showing up in the outbox. Returns the just-sent attempts."""
    teachers_known = {t["id"] for t in list_teachers(user_id)}
    if teacher_id not in teachers_known:
        raise ValueError(f"unknown teacher_id: {teacher_id!r}")

    all_attempts = list_attempts(user_id)
    latest_unsent = _latest_by_problem(
        [a for a in all_attempts if a.get("sent_at") is None]
    )
    # Single timestamp shared by every attempt in this submission so the
    # UI can group by `sent_at` to reconstruct it.
    stamp = _now()
    sent_now: list[dict] = []
    for a in all_attempts:
        if a.get("sent_at") is not None:
            continue
        is_latest = latest_unsent.get(a["problem_id"], {}).get("id") == a["id"]
        if is_latest:
            a["sent_to"] = [teacher_id]
            sent_now.append(a)
        a["sent_at"] = stamp
        _attempt_path(user_id, a["id"]).write_text(json.dumps(a, indent=2))
    return sent_now


# --- submissions (groups of attempts sharing sent_at) ---


def list_submissions(user_id: str) -> list[dict]:
    """All submissions, newest first. Each:
        {
          "sent_at": str,
          "teacher_id": str,
          "attempts": [normalized attempt dicts...],
          "state": "pending" | "returned" | "acked",
        }
    Superseded attempts (sent with empty `sent_to`) are excluded.
    """
    by_sent: dict[str, list[dict]] = {}
    for a in list_attempts(user_id):
        ts = a.get("sent_at")
        if not ts or not a.get("sent_to"):
            continue
        by_sent.setdefault(ts, []).append(a)

    out: list[dict] = []
    for ts, lst in by_sent.items():
        teacher_id = lst[0]["sent_to"][0]
        all_reviewed = all(
            teacher_id in (a.get("reviews") or {}) for a in lst
        )
        all_acked = all(a.get("acked_at") for a in lst)
        if not all_reviewed:
            state = "pending"
        elif not all_acked:
            state = "returned"
        else:
            state = "acked"
        out.append({
            "sent_at": ts,
            "teacher_id": teacher_id,
            "attempts": lst,
            "state": state,
        })
    out.sort(key=lambda s: s["sent_at"], reverse=True)
    return out


def ack_submission(user_id: str, sent_at: str) -> int:
    """Mark every reviewed attempt in the submission as acked. Returns
    the number of attempts updated. Unreviewed attempts are left alone
    (acking again later is a no-op for them)."""
    stamp = _now()
    count = 0
    for a in list_attempts(user_id):
        if a.get("sent_at") != sent_at:
            continue
        if not a.get("sent_to"):
            continue
        teacher_id = a["sent_to"][0]
        if teacher_id not in (a.get("reviews") or {}):
            continue
        if a.get("acked_at"):
            continue
        a["acked_at"] = stamp
        _attempt_path(user_id, a["id"]).write_text(json.dumps(a, indent=2))
        count += 1
    return count


# --- teacher-side queues (parameterized by teacher_id) ---


def pending_for_teacher(user_id: str, teacher_id: str) -> list[dict]:
    """Latest attempt per problem that this teacher is asked to grade.

    "Latest" = most recently submitted attempt for that problem whose
    `sent_to` includes this teacher and whose `reviews` does not yet
    have a verdict from this teacher."""
    visible = [
        a for a in list_attempts(user_id)
        if teacher_id in a.get("sent_to", [])
        and teacher_id not in a.get("reviews", {})
    ]
    by_pid: dict[str, dict] = {}
    for a in visible:
        pid = a["problem_id"]
        prev = by_pid.get(pid)
        if prev is None or a["submitted_at"] > prev["submitted_at"]:
            by_pid[pid] = a
    out = list(by_pid.values())
    out.sort(key=lambda d: d["submitted_at"])
    return out


def reviewed_by_teacher(user_id: str, teacher_id: str) -> list[dict]:
    """Every attempt this teacher has reviewed, newest verdict first.

    Not deduped per problem — a student may submit a problem more than
    once after correcting an earlier attempt, and the history view wants
    to surface each verdict in its own submission group."""
    reviewed = [
        a for a in list_attempts(user_id)
        if teacher_id in a.get("reviews", {})
    ]
    reviewed.sort(
        key=lambda d: d["reviews"][teacher_id].get("reviewed_at", ""),
        reverse=True,
    )
    return reviewed


def set_review(
    user_id: str, teacher_id: str, attempt_id: str, verdict: str,
) -> dict | None:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict: {verdict!r}")
    a = load_attempt(user_id, attempt_id)
    if a is None:
        return None
    if teacher_id not in a.get("sent_to", []):
        # Defensive: a teacher can only grade attempts addressed to them.
        raise PermissionError("attempt was not sent to this teacher")
    a["reviews"][teacher_id] = {"verdict": verdict, "reviewed_at": _now()}
    _attempt_path(user_id, attempt_id).write_text(json.dumps(a, indent=2))
    return a


def problem_statuses(user_id: str) -> dict[str, dict]:
    """Per-problem study status, keyed by problem id. Only problems with
    at least one attempt appear in the map. `last_verdict` is the verdict
    of the most recent review on the latest attempt (across teachers),
    or None if the latest attempt has no reviews yet."""
    out: dict[str, dict] = {}
    for pid, a in _latest_by_problem(list_attempts(user_id)).items():
        reviews = a.get("reviews", {}) or {}
        last_verdict: str | None = None
        latest_at = ""
        for r in reviews.values():
            ts = r.get("reviewed_at", "")
            if ts > latest_at:
                latest_at = ts
                last_verdict = r.get("verdict")
        out[pid] = {"last_verdict": last_verdict}
    return out


def reviewed_attempts(user_id: str) -> list[dict]:
    """Student-side graded history: every reviewed attempt the student
    has acknowledged, newest verdict first.

    Not deduped per problem — when the student re-attempts a problem
    after an "incorrect" verdict, both the earlier (incorrect) and later
    (correct) attempts should appear in their respective submission
    groups so the progression is visible. Unacked submissions live on
    the home page's in-flight panel until the student marks them read."""
    def _latest_review(a: dict) -> str:
        return max(
            (r.get("reviewed_at", "") for r in a.get("reviews", {}).values()),
            default="",
        )

    out = [
        a for a in list_attempts(user_id)
        if a.get("reviews") and a.get("acked_at")
    ]
    out.sort(key=_latest_review, reverse=True)
    return out


# --- teachers (capability URLs) ---


def list_teachers(user_id: str) -> list[dict]:
    tdir = teachers_dir(user_id)
    if not tdir.exists():
        return []
    out: list[dict] = []
    for p in tdir.glob("teacher_*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    out.sort(key=lambda t: t.get("created_at", ""))
    return out


def create_teacher(user_id: str, label: str) -> dict:
    tdir = teachers_dir(user_id)
    tdir.mkdir(parents=True, exist_ok=True)
    tid = f"teacher_{secrets.token_hex(4)}"
    rec = {
        "id": tid,
        "token": f"tt_{secrets.token_urlsafe(24)}",
        "label": label.strip() or "Teacher",
        "created_at": _now(),
    }
    teacher_path(user_id, tid).write_text(json.dumps(rec, indent=2))
    return rec


def get_teacher(user_id: str, teacher_id: str) -> dict | None:
    p = teacher_path(user_id, teacher_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def delete_teacher(user_id: str, teacher_id: str) -> bool:
    p = teacher_path(user_id, teacher_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def regenerate_teacher_token(user_id: str, teacher_id: str) -> dict | None:
    rec = get_teacher(user_id, teacher_id)
    if rec is None:
        return None
    rec["token"] = f"tt_{secrets.token_urlsafe(24)}"
    teacher_path(user_id, teacher_id).write_text(json.dumps(rec, indent=2))
    return rec


def update_teacher_label(user_id: str, teacher_id: str, label: str) -> dict | None:
    rec = get_teacher(user_id, teacher_id)
    if rec is None:
        return None
    rec["label"] = label.strip() or "Teacher"
    teacher_path(user_id, teacher_id).write_text(json.dumps(rec, indent=2))
    return rec


def resolve_token(token: str) -> tuple[str, dict] | None:
    """Resolve a token to (user_id, teacher_record). Globs across user
    dirs (no shared registry); O(num_teachers), trivial at our scale."""
    if not ATTEMPTS_ROOT.exists():
        return None
    for p in ATTEMPTS_ROOT.glob("*/teachers/teacher_*.json"):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if rec.get("token") == token:
            return p.parent.parent.name, rec
    return None
