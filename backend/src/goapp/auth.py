"""User identity from Google IAP headers.

In production, Cloud Run sits behind Identity-Aware Proxy. IAP authenticates
the caller against Google and forwards the verified email in the
`X-Goog-Authenticated-User-Email` header. The header value looks like
`accounts.google.com:user@example.com`; we strip the prefix and hash it to
a short stable id usable as a filesystem path component.

When the header is absent (local dev, direct curl), we fall back to a
fixed `local` id so single-user development just works.
"""

from __future__ import annotations

import hashlib

from fastapi import Request

LOCAL_USER_ID = "local"
IAP_HEADER = "x-goog-authenticated-user-email"


def _hash_email(email: str) -> str:
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]


def _email_from_request(request: Request) -> str | None:
    raw = request.headers.get(IAP_HEADER)
    if not raw:
        return None
    email = raw.split(":", 1)[-1].strip().lower()
    return email or None


def user_id_from_request(request: Request) -> str:
    """Resolve the IAP-authenticated caller's user_id.

    Side effect: when we see an email for the first time on this account,
    record it in the user's profile so the rest of the app can show it
    next to their display_name. We only write when something would
    actually change, so steady-state requests don't touch storage.
    """
    email = _email_from_request(request)
    if email is None:
        return LOCAL_USER_ID
    user_id = _hash_email(email)
    # Lazy import to avoid a top-level cycle (profile imports paths,
    # which is benign, but keep auth.py free of profile-side concerns).
    from .profile import load_profile, save_profile
    if load_profile(user_id).get("email") != email:
        save_profile(user_id, email=email)
    return user_id
