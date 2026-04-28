"""Local PDF ingest CLI: detect boards on the laptop, then push to GCS.

Cloud Run does CPU-only ONNX inference and chews minutes per book; running
this locally is much faster (and free). Output is identical to a
server-side ingest, so the cloud app can't tell the difference.

Usage:
    python -m goapp.cli.ingest_pdf --pdf book.pdf --user pwsiegel@gmail.com

If --user looks like an email it's hashed via `_hash_email` to match the
production IAP-derived user_id. Otherwise it's treated as the raw id.
After detection completes, files under
$GOAPP_DATA_DIR/data/tsumego/{user_id}/ are rsynced to the GCS bucket
(--no-upload to skip).

The detection pipeline (and its multiprocess fan-out) lives in
`goapp.pdf_ingest` and is shared with the API ingest job runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .. import pdf_ingest
from ..auth import _hash_email
from ..paths import tsumego_dir


def resolve_user_id(user: str) -> str:
    if "@" in user:
        return _hash_email(user.strip().lower())
    return user


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run board detection on a PDF locally and sync results to GCS.",
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument(
        "--user", required=True,
        help="email (gets hashed) or raw user_id",
    )
    parser.add_argument(
        "--source", default=None,
        help="source name to record (defaults to PDF filename)",
    )
    parser.add_argument("--bucket", default="tsumego-pwsiegel-data")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=pdf_ingest.default_workers(),
        help=f"page-level worker processes (default: {pdf_ingest.default_workers()})",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"error: pdf not found: {args.pdf}", file=sys.stderr)
        return 1

    user_id = resolve_user_id(args.user)
    source = args.source or args.pdf.name
    pdf_path = str(args.pdf.resolve())

    print(f"==> ingesting {args.pdf.name} as user_id={user_id} source={source!r}")
    print(f"  workers: {args.workers}")

    total_saved = 0
    skipped = 0
    t_start = time.time()
    for line in pdf_ingest.iter_ingest_events(
        pdf_path, source, user_id, workers=args.workers,
    ):
        ev = json.loads(line)
        kind = ev.get("event")
        if kind == "start":
            print(f"  pages: {ev['total_pages']}")
        elif kind == "page_rendered":
            print(f"  [{ev['page']}/{ev['total_pages']}] page complete")
        elif kind == "board_saved":
            total_saved = ev["total_saved"]
        elif kind == "done":
            total_saved = ev["total_saved"]
            skipped = ev["skipped"]
    print(f"  detection took {time.time() - t_start:.1f}s")
    print(f"==> detection done: {total_saved} saved, {skipped} skipped")

    if args.no_upload:
        print("==> skipping upload (--no-upload)")
        return 0

    udir = tsumego_dir(user_id)
    if not udir.exists():
        print(f"==> nothing to upload (no problems for {user_id})")
        return 0

    dst = f"gs://{args.bucket}/data/tsumego/{user_id}/"
    print(f"==> syncing {udir}/ -> {dst}")
    rc = subprocess.call(["gsutil", "-m", "rsync", "-r", f"{udir}/", dst])
    if rc != 0:
        print(f"error: gsutil rsync exited {rc}", file=sys.stderr)
        return rc
    print("==> upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
