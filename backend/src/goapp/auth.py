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


def user_id_from_request(request: Request) -> str:
    raw = request.headers.get(IAP_HEADER)
    if not raw:
        return LOCAL_USER_ID
    email = raw.split(":", 1)[-1].strip().lower()
    if not email:
        return LOCAL_USER_ID
    return _hash_email(email)
