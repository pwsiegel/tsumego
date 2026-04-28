"""Manually create a student → teacher link.

Phase 1 has no invite/accept UX, so we bootstrap the two-user relationship
from the shell:

    python -m goapp.cli.link --student a@x.com --teacher b@x.com

Both arguments accept either an email (auto-hashed to match the IAP-derived
user_id) or a raw 16-char user_id. Pass --remove to undo a link.
"""

from __future__ import annotations

import argparse
import sys

from ..auth import _hash_email
from ..links import add_teacher, list_teachers, remove_teacher


def resolve_user_id(s: str) -> str:
    s = s.strip()
    if "@" in s:
        return _hash_email(s.lower())
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Create or remove a student → teacher link.")
    p.add_argument("--student", required=True, help="email (gets hashed) or raw user_id")
    p.add_argument("--teacher", required=True, help="email (gets hashed) or raw user_id")
    p.add_argument("--remove", action="store_true", help="remove the link instead of adding it")
    args = p.parse_args()

    student_uid = resolve_user_id(args.student)
    teacher_uid = resolve_user_id(args.teacher)
    print(f"student: {student_uid}")
    print(f"teacher: {teacher_uid}")

    if args.remove:
        ok = remove_teacher(student_uid, teacher_uid)
        print(f"removed link" if ok else "no such link")
    else:
        try:
            added = add_teacher(student_uid, teacher_uid)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print("added link" if added else "link already present")

    print(f"current teachers for student: {list_teachers(student_uid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
