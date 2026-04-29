"""One-off: replace the hm2.pdf collection on GCP with the post-split version.

Wipes every problem on GCP whose JSON has source=hm2.pdf (strict scan —
downloads every .json under the user's prod dir and filters by content,
so testing-only hm2 entries that exist only on GCP also get cleaned up),
mirrors the same wipe in the local prod-mirror dir, then copies the
post-split hm2 problems from a source user dir (default: 'local') into
both the local prod dir and GCP.

Usage:
    python -m goapp.cli.push_hm2                    # dry run
    python -m goapp.cli.push_hm2 --apply            # commit

Assumes `gsutil` is on PATH and authed for the target bucket.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..paths import tsumego_dir

SOURCE = "hm2.pdf"


def _local_triples(udir: Path, source: str) -> list[tuple[Path, Path, Path | None]]:
    """All (json, sgf, png) triples in udir whose JSON source matches."""
    triples: list[tuple[Path, Path, Path | None]] = []
    for jp in sorted(udir.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("source") != source:
            continue
        sgf = jp.with_suffix(".sgf")
        png = (udir / d["image"]) if d.get("image") else None
        triples.append((jp, sgf, png))
    return triples


def _gcp_hm2_files(gs_prefix: str, source: str) -> list[str]:
    """List GCS object URIs for files belonging to `source` in `gs_prefix/`.

    Strategy: bulk-download every .json under gs_prefix/ to a temp dir,
    parse each, and for matching ones return [json, sgf, png] URIs.
    Bulk cp is vastly faster than per-object gsutil cat.
    """
    with tempfile.TemporaryDirectory(prefix="hm2_scan_") as tmp:
        tmp_path = Path(tmp)
        print(f"  bulk-downloading *.json from {gs_prefix} ...")
        rc = subprocess.call([
            "gsutil", "-m", "-q", "cp",
            f"{gs_prefix.rstrip('/')}/*.json",
            f"{tmp_path}/",
        ])
        if rc != 0:
            print(f"error: gsutil cp exited {rc}", file=sys.stderr)
            raise SystemExit(rc)

        uris: list[str] = []
        for jp in tmp_path.glob("*.json"):
            try:
                d = json.loads(jp.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("source") != source:
                continue
            stem = jp.stem
            uris.append(f"{gs_prefix.rstrip('/')}/{stem}.json")
            uris.append(f"{gs_prefix.rstrip('/')}/{stem}.sgf")
            if d.get("image"):
                uris.append(f"{gs_prefix.rstrip('/')}/{d['image']}")
        return uris


def _gsutil_rm(uris: list[str]) -> None:
    if not uris:
        return
    # Pipe via -I so we don't blow the argv limit.
    p = subprocess.Popen(
        ["gsutil", "-m", "rm", "-I"], stdin=subprocess.PIPE, text=True,
    )
    assert p.stdin is not None
    p.stdin.write("\n".join(uris) + "\n")
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        print(f"error: gsutil rm exited {rc}", file=sys.stderr)
        raise SystemExit(rc)


def _gsutil_cp_to(files: list[Path], gs_prefix: str) -> None:
    """Bulk-upload `files` to `gs_prefix/`.

    NOTE: `gsutil cp -I` is broken on this system — it reads only the
    first 2 lines from stdin and exits. Passing files as positional argv
    works correctly. argv stays well under ARG_MAX even at thousands of
    paths, but we chunk to be defensive.
    """
    if not files:
        return
    CHUNK = 500
    dst = gs_prefix.rstrip("/") + "/"
    for i in range(0, len(files), CHUNK):
        batch = [str(f) for f in files[i:i + CHUNK]]
        rc = subprocess.call(["gsutil", "-m", "cp", *batch, dst])
        if rc != 0:
            print(f"error: gsutil cp exited {rc}", file=sys.stderr)
            raise SystemExit(rc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default="tsumego-pwsiegel-data")
    parser.add_argument(
        "--from-user-id", default="local",
        help="local user dir holding the post-split hm2 (source of new files)",
    )
    parser.add_argument(
        "--to-user-id", default="7a692608e710b808",
        help="prod user dir to overwrite (mirror + GCP)",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    src_dir = tsumego_dir(args.from_user_id)
    dst_dir = tsumego_dir(args.to_user_id)
    gs_prefix = f"gs://{args.bucket}/data/tsumego/{args.to_user_id}"

    if not src_dir.exists():
        print(f"error: source dir missing: {src_dir}", file=sys.stderr)
        return 1
    if not dst_dir.exists():
        print(f"error: dst dir missing: {dst_dir}", file=sys.stderr)
        return 1

    # New files to upload — the post-split hm2 in src_dir.
    new_triples = _local_triples(src_dir, SOURCE)
    if not new_triples:
        print(f"error: no {SOURCE} problems in {src_dir}", file=sys.stderr)
        return 1
    new_files: list[Path] = []
    seen: set[Path] = set()
    for jp, sgf, png in new_triples:
        for f in (jp, sgf, png):
            if f and f not in seen:
                seen.add(f)
                new_files.append(f)
    print(f"new files to upload: {len(new_files)} ({len(new_triples)} problems)")

    # Old files to wipe locally (prod-mirror dir).
    old_local_triples = _local_triples(dst_dir, SOURCE)
    old_local_files: list[Path] = []
    seen2: set[Path] = set()
    for jp, sgf, png in old_local_triples:
        for f in (jp, sgf, png):
            if f and f not in seen2:
                seen2.add(f)
                old_local_files.append(f)
    print(f"old files in local prod-mirror to delete: {len(old_local_files)}"
          f" ({len(old_local_triples)} problems)")

    # Old files to wipe on GCP (strict scan).
    print(f"scanning {gs_prefix}/ for {SOURCE!r} files ...")
    old_gcp_uris = _gcp_hm2_files(gs_prefix, SOURCE)
    print(f"old files on GCP to delete: {len(old_gcp_uris)}"
          f" (≈{len(old_gcp_uris)//3} problems)")

    if not args.apply:
        print("\ndry run — pass --apply to commit")
        return 0

    print("\n=== applying ===")

    print("[1/4] gsutil rm old hm2 on GCP ...")
    _gsutil_rm(old_gcp_uris)

    print("[2/4] deleting old hm2 in local prod-mirror ...")
    for f in old_local_files:
        f.unlink(missing_ok=True)

    print("[3/4] copying new hm2 into local prod-mirror ...")
    for f in new_files:
        shutil.copy2(f, dst_dir / f.name)

    print("[4/4] gsutil cp new hm2 to GCP ...")
    # Upload from the destination dir (where we just placed copies) so
    # filenames on GCP match the local prod-mirror exactly.
    files_to_upload = [dst_dir / f.name for f in new_files]
    _gsutil_cp_to(files_to_upload, gs_prefix)

    print(f"\ndone: replaced {len(old_local_triples)} → {len(new_triples)}"
          f" hm2 problems; GCP wiped {len(old_gcp_uris)} files,"
          f" uploaded {len(files_to_upload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
