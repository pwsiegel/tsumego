"""Cloud Storage helpers (signed URLs).

Used only when running in cloud mode (GOAPP_GCS_BUCKET is set). Locally,
all storage goes through the filesystem.

Signing relies on IAM SignBlob — the runtime SA needs
`roles/iam.serviceAccountTokenCreator` on itself so it can sign on its
own behalf without a downloaded key file.
"""

from __future__ import annotations

import os
from datetime import timedelta

import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

_BUCKET_ENV = "GOAPP_GCS_BUCKET"


def is_enabled() -> bool:
    return bool(os.environ.get(_BUCKET_ENV))


def bucket_name() -> str:
    name = os.environ.get(_BUCKET_ENV)
    if not name:
        raise RuntimeError(f"{_BUCKET_ENV} not set")
    return name


def signed_upload_url(
    object_key: str,
    *,
    content_type: str = "application/pdf",
    expires_minutes: int = 15,
) -> str:
    credentials, _ = google.auth.default()
    if not credentials.valid:
        credentials.refresh(Request())
    client = storage.Client(credentials=credentials)
    blob = client.bucket(bucket_name()).blob(object_key)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expires_minutes),
        method="PUT",
        content_type=content_type,
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )
