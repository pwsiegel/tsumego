"""Solve attempts, batches, and per-reviewer reviews.

Layout under attempts_dir(user_id):
    attempt_*.json       one attempt per file
    links.json           {"teachers": [<user_id>, ...]}

Attempt schema:
    {
      "id": "attempt_<ts>_<rand>",
      "problem_id": "tsumego_...",
      "moves": [{"col": int, "row": int}, ...],   ordered, may repeat
      "submitted_at": "<UTC ISO>",
      "sent_to": ["<reviewer_user_id>"],   single recipient (one-element
                                            list for storage compatibility)
      "sent_at": "<UTC ISO>" | null, when the attempt was sent
      "reviews": { "<reviewer_user_id>": {"verdict": "...", "reviewed_at": "..."} },
      "acked_at": "<UTC ISO>" | null,  when the student marked the
                                       reviewed submission as read
    }

Submissions are groups of attempts that share `sent_at`. Each submission
goes to exactly one reviewer (a linked teacher's user_id). State for a
submission:
    pending  — at least one attempt has no review from the reviewer yet
    returned — all attempts reviewed, but at least one not yet acked
    acked    — student has marked the submission as read

Authorization is via `goapp.links`: a reviewer may only see / grade
attempts owned by a student who has linked them.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .links import is_teacher_of
from .paths import attempts_dir

VERDICTS = ("correct", "incorrect")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _attempt_path(user_id: str, attempt_id: str) -> Path:
    return attempts_dir(user_id) / f"{attempt_id}.json"


def _normalize_attempt(a: dict) -> dict:
    out = dict(a)
    out.setdefault("sent_to", [])
    out.setdefault("sent_at", None)
    out.setdefault("acked_at", None)
    out.setdefault("reviews", {})
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
    student would push on the next 'send to teacher' action."""
    unsent = [a for a in list_attempts(user_id) if a.get("sent_at") is None]
    out = list(_latest_by_problem(unsent).values())
    out.sort(key=lambda d: d.get("submitted_at", ""))
    return out


def remove_from_batch(user_id: str, problem_id: str) -> int:
    """Drop every unsent attempt for `problem_id` from the student's
    batch. Sent attempts are left alone (they belong to a submission)."""
    removed = 0
    for a in list_attempts(user_id):
        if a.get("problem_id") != problem_id or a.get("sent_at") is not None:
            continue
        path = _attempt_path(user_id, a["id"])
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def send_to_reviewer(student_uid: str, reviewer_uid: str) -> list[dict]:
    """Send all unsent attempts (latest per problem) to a single reviewer
    as one submission. Earlier unsent attempts on the same problem are
    superseded — they get marked sent with no recipient so they don't
    keep showing up in the outbox. Returns the just-sent attempts."""
    if not is_teacher_of(reviewer_uid, student_uid):
        raise ValueError(f"{reviewer_uid!r} is not a linked teacher of {student_uid!r}")

    all_attempts = list_attempts(student_uid)
    latest_unsent = _latest_by_problem(
        [a for a in all_attempts if a.get("sent_at") is None]
    )
    stamp = _now()
    sent_now: list[dict] = []
    for a in all_attempts:
        if a.get("sent_at") is not None:
            continue
        is_latest = latest_unsent.get(a["problem_id"], {}).get("id") == a["id"]
        if is_latest:
            a["sent_to"] = [reviewer_uid]
            sent_now.append(a)
        a["sent_at"] = stamp
        _attempt_path(student_uid, a["id"]).write_text(json.dumps(a, indent=2))
    return sent_now


# --- submissions (groups of attempts sharing sent_at) ---


def list_submissions(user_id: str) -> list[dict]:
    """All submissions, newest first. Each:
        {
          "sent_at": str,
          "reviewer_id": str,
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
        reviewer_id = lst[0]["sent_to"][0]
        all_reviewed = all(
            reviewer_id in (a.get("reviews") or {}) for a in lst
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
            "reviewer_id": reviewer_id,
            "attempts": lst,
            "state": state,
        })
    out.sort(key=lambda s: s["sent_at"], reverse=True)
    return out


def ack_submission(user_id: str, sent_at: str) -> int:
    """Mark every reviewed attempt in the submission as acked. Returns
    the number of attempts updated."""
    stamp = _now()
    count = 0
    for a in list_attempts(user_id):
        if a.get("sent_at") != sent_at:
            continue
        if not a.get("sent_to"):
            continue
        reviewer_id = a["sent_to"][0]
        if reviewer_id not in (a.get("reviews") or {}):
            continue
        if a.get("acked_at"):
            continue
        a["acked_at"] = stamp
        _attempt_path(user_id, a["id"]).write_text(json.dumps(a, indent=2))
        count += 1
    return count


# --- teacher-side queues (parameterized by reviewer user_id) ---


def pending_for_reviewer(student_uid: str, reviewer_uid: str) -> list[dict]:
    """Latest attempt per problem that this reviewer is asked to grade."""
    visible = [
        a for a in list_attempts(student_uid)
        if reviewer_uid in a.get("sent_to", [])
        and reviewer_uid not in a.get("reviews", {})
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


def reviewed_by_reviewer(student_uid: str, reviewer_uid: str) -> list[dict]:
    """Every attempt this reviewer has reviewed for the student, newest
    verdict first. Not deduped per problem."""
    reviewed = [
        a for a in list_attempts(student_uid)
        if reviewer_uid in a.get("reviews", {})
    ]
    reviewed.sort(
        key=lambda d: d["reviews"][reviewer_uid].get("reviewed_at", ""),
        reverse=True,
    )
    return reviewed


def set_review(
    student_uid: str, reviewer_uid: str, attempt_id: str, verdict: str,
) -> dict | None:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict: {verdict!r}")
    a = load_attempt(student_uid, attempt_id)
    if a is None:
        return None
    if reviewer_uid not in a.get("sent_to", []):
        raise PermissionError("attempt was not sent to this reviewer")
    a["reviews"][reviewer_uid] = {"verdict": verdict, "reviewed_at": _now()}
    _attempt_path(student_uid, attempt_id).write_text(json.dumps(a, indent=2))
    return a


def problem_statuses(user_id: str) -> dict[str, dict]:
    """Per-problem study status, keyed by problem id."""
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
        out[pid] = {
            "last_verdict": last_verdict,
            "latest_attempt_at": a.get("submitted_at"),
        }
    return out


def reviewed_attempts(user_id: str) -> list[dict]:
    """Student-side graded history: every reviewed attempt the student
    has acknowledged, newest verdict first."""
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
