"""Per-user profile data.

Stores `display_name` (what other users see) and `email` (the IAP-verified
address used to derive the user_id). Email is auto-recorded on first
authenticated request and from the link CLI at bootstrap. The file lives
next to the user's attempts/teachers so it's part of the same per-user
blob and gets cleaned up the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import attempts_dir


def _profile_path(user_id: str) -> Path:
    return attempts_dir(user_id) / "profile.json"


def load_profile(user_id: str) -> dict:
    p = _profile_path(user_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_profile(user_id: str, **fields) -> dict:
    udir = attempts_dir(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    current = load_profile(user_id)
    current.update(fields)
    _profile_path(user_id).write_text(json.dumps(current, indent=2))
    return current


def display_name(user_id: str) -> str:
    """Configured display name, or the raw user_id if none has been set."""
    return load_profile(user_id).get("display_name") or user_id


def email_for(user_id: str) -> str | None:
    return load_profile(user_id).get("email") or None
