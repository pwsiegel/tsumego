"""One-off: backfill `problem_snapshot` on already-sent attempts.

Once `send_to_reviewer` started writing a frozen problem snapshot into
each sent attempt (so submission/teacher views survive later edits or
deletions to the underlying problem), pre-feature attempts still have no
snapshot. This script walks the user's attempts dir, fills in the
snapshot from the live problem record for every attempt that has
`sent_at` set, and (optionally) syncs the rewritten files up to GCS.

Usage:
    python -m goapp.cli.backfill_attempt_snapshots                          # dry run, local-only
    python -m goapp.cli.backfill_attempt_snapshots --pull                   # dry run, pull GCS first
    python -m goapp.cli.backfill_attempt_snapshots --pull --apply --push    # full prod round-trip

Defaults to user_id=7a692608e710b808 (prod). Pass --user-id for others.

Prod attempts live only on GCS; the local attempts dir for the prod
user is empty in dev. `--pull` rsyncs GCS → local first; `--push`
uploads the modified files back after `--apply`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..paths import attempts_dir
from ..study import make_problem_snapshot
from ..tsumego import load_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user-id", default="7a692608e710b808")
    parser.add_argument("--bucket", default="tsumego-pwsiegel-data")
    parser.add_argument("--apply", action="store_true",
                        help="write changes (default: dry run)")
    parser.add_argument("--pull", action="store_true",
                        help="rsync GCS attempts → local before scanning")
    parser.add_argument("--push", action="store_true",
                        help="also push modified files to GCS (requires --apply)")
    args = parser.parse_args()

    udir = attempts_dir(args.user_id)
    gs_prefix = f"gs://{args.bucket}/data/attempts/{args.user_id}"

    if args.pull:
        udir.mkdir(parents=True, exist_ok=True)
        print(f"rsync {gs_prefix}/ → {udir}/ ...")
        rc = subprocess.call([
            "gsutil", "-m", "rsync", "-r",
            gs_prefix.rstrip("/") + "/",
            str(udir) + "/",
        ])
        if rc != 0:
            print(f"error: gsutil rsync exited {rc}", file=sys.stderr)
            return rc

    if not udir.exists():
        print(f"error: attempts dir missing: {udir}", file=sys.stderr)
        return 1

    sent_total = 0
    already = 0
    to_fill: list[tuple[Path, dict, dict]] = []  # (path, attempt, snapshot)
    missing_problem: list[tuple[Path, str]] = []

    for jp in sorted(udir.glob("attempt_*.json")):
        try:
            a = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if not a.get("sent_at"):
            continue
        sent_total += 1
        if a.get("problem_snapshot"):
            already += 1
            continue
        pid = a.get("problem_id")
        p = load_problem(args.user_id, pid) if pid else None
        if p is None:
            missing_problem.append((jp, pid or "<no problem_id>"))
            continue
        to_fill.append((jp, a, make_problem_snapshot(p)))

    print(f"sent attempts: {sent_total}")
    print(f"  already snapshotted: {already}")
    print(f"  to backfill:         {len(to_fill)}")
    print(f"  problem missing:     {len(missing_problem)}")
    for jp, pid in missing_problem[:10]:
        print(f"    {jp.name}: problem {pid!r} not in tsumego dir")
    if len(missing_problem) > 10:
        print(f"    ... and {len(missing_problem) - 10} more")

    if not args.apply:
        print("\ndry run — pass --apply to write")
        return 0

    print("\nwriting snapshots ...")
    written: list[Path] = []
    for jp, a, snap in to_fill:
        a["problem_snapshot"] = snap
        jp.write_text(json.dumps(a, indent=2))
        written.append(jp)
    print(f"wrote {len(written)} files")

    if args.push and written:
        print(f"\nuploading {len(written)} files to {gs_prefix}/ ...")
        CHUNK = 500
        dst = gs_prefix + "/"
        for i in range(0, len(written), CHUNK):
            batch = [str(f) for f in written[i:i + CHUNK]]
            rc = subprocess.call(["gsutil", "-m", "cp", *batch, dst])
            if rc != 0:
                print(f"error: gsutil cp exited {rc}", file=sys.stderr)
                return rc
        print("uploaded.")
    elif args.push:
        print("nothing to upload.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
