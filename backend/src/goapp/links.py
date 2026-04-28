"""Student ↔ teacher links.

A "link" is a directed edge: student grants a teacher permission to read
their submissions and post verdicts. The link is stored on the student
side, in `attempts/{student_uid}/links.json`:

    {"teachers": ["<teacher_uid>", ...]}

Querying the inverse — "which students linked me as a teacher?" — scans
all users' link files. O(num_users); fine at single-digit scale.

Linking is purely manual today (`python -m goapp.cli.link …`). A future
iteration can layer an invite/accept UX on top of these primitives.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import ATTEMPTS_ROOT, attempts_dir


def _links_path(user_id: str) -> Path:
    return attempts_dir(user_id) / "links.json"


def _load(user_id: str) -> dict:
    p = _links_path(user_id)
    if not p.exists():
        return {"teachers": []}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"teachers": []}
    d.setdefault("teachers", [])
    return d


def _save(user_id: str, data: dict) -> None:
    udir = attempts_dir(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    _links_path(user_id).write_text(json.dumps(data, indent=2))


def list_teachers(student_uid: str) -> list[str]:
    """User_ids the student has granted teacher access to."""
    return list(_load(student_uid)["teachers"])


def add_teacher(student_uid: str, teacher_uid: str) -> bool:
    """Idempotent. Returns True if newly added, False if already present.

    Self-links are allowed: a single dev-mode user can play both sides for
    local testing.
    """
    d = _load(student_uid)
    if teacher_uid in d["teachers"]:
        return False
    d["teachers"].append(teacher_uid)
    _save(student_uid, d)
    return True


def remove_teacher(student_uid: str, teacher_uid: str) -> bool:
    d = _load(student_uid)
    if teacher_uid not in d["teachers"]:
        return False
    d["teachers"].remove(teacher_uid)
    _save(student_uid, d)
    return True


def is_teacher_of(teacher_uid: str, student_uid: str) -> bool:
    return teacher_uid in _load(student_uid)["teachers"]


def students_of(teacher_uid: str) -> list[str]:
    """Every student who has linked this teacher. Scans all user link
    files — fine at single-digit user counts."""
    if not ATTEMPTS_ROOT.exists():
        return []
    out: list[str] = []
    for p in ATTEMPTS_ROOT.glob("*/links.json"):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if teacher_uid in (d.get("teachers") or []):
            out.append(p.parent.name)
    return out
