"""Export a hand-annotated collection as a versioned validation dataset.

Copies every problem (SGF + metadata JSON + crop PNG) belonging to a
single source PDF into a self-contained directory with a manifest,
renamed to a stable source_board_idx-based scheme so future validation
runs can reference problems by a sortable human-readable ID.

Output layout:
    <out_dir>/
        manifest.json         summary + per-problem list
        README.md             human notes
        images/<prefix>_NNNN.png
        metadata/<prefix>_NNNN.json
        sgf/<prefix>_NNNN.sgf

Usage:
    uv --directory api run python -m goapp_api.export_val_dataset \\
        --source hm2.pdf \\
        --out ~/data/go-app/data/val/hm2 \\
        --prefix hm2
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .paths import TSUMEGO_DIR


def export(source: str, out_dir: Path, prefix: str) -> dict:
    if not TSUMEGO_DIR.exists():
        raise SystemExit(f"no tsumego dir at {TSUMEGO_DIR}")

    # Collect every problem belonging to `source`, sorted by source_board_idx
    # so filenames are in reading order.
    problems: list[dict] = []
    for mp in TSUMEGO_DIR.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("source") != source:
            continue
        problems.append(d)
    if not problems:
        raise SystemExit(f"no problems found for source={source!r}")
    problems.sort(key=lambda d: d.get("source_board_idx", 0))

    # Recreate output tree.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "metadata").mkdir(parents=True)
    (out_dir / "sgf").mkdir(parents=True)

    # Zero-pad width picked from the max source_board_idx we see.
    max_idx = max(p.get("source_board_idx", 0) for p in problems)
    pad = max(4, len(str(max_idx)))

    manifest_entries: list[dict] = []
    counts = {"accepted": 0, "accepted_edited": 0, "rejected": 0, "unreviewed": 0}

    for p in problems:
        idx = p["source_board_idx"]
        pid = p["id"]
        stem = f"{prefix}_{str(idx).zfill(pad)}"

        # Copy image / metadata / SGF.
        src_image = TSUMEGO_DIR / (p.get("image") or "")
        if p.get("image") and src_image.exists():
            shutil.copy2(src_image, out_dir / "images" / f"{stem}.png")
        src_sgf = TSUMEGO_DIR / f"{pid}.sgf"
        if src_sgf.exists():
            shutil.copy2(src_sgf, out_dir / "sgf" / f"{stem}.sgf")
        # Metadata: copy but update the embedded filename fields to match
        # the new stem so downstream tools can load the package without
        # needing the manifest.
        new_meta = {
            **p,
            "export_stem": stem,
            "image": f"{stem}.png" if p.get("image") else None,
            "sgf": f"{stem}.sgf",
        }
        (out_dir / "metadata" / f"{stem}.json").write_text(
            json.dumps(new_meta, indent=2)
        )

        status = p.get("status", "unreviewed")
        counts[status] = counts.get(status, 0) + 1

        manifest_entries.append({
            "stem": stem,
            "id": pid,
            "source_board_idx": idx,
            "status": status,
            "n_stones": len(p.get("stones", [])),
            "image": f"images/{stem}.png" if p.get("image") else None,
            "metadata": f"metadata/{stem}.json",
            "sgf": f"sgf/{stem}.sgf",
        })

    manifest = {
        "source": source,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(problems),
        "counts": counts,
        "problems": manifest_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Human-readable notes.
    readme = [
        f"# Validation dataset: {source}",
        "",
        f"Exported from `~/data/go-app/data/tsumego/` on {manifest['exported_at']}.",
        "",
        f"**Total**: {len(problems)} problems",
        "",
        "| status | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items() if v > 0],
        "",
        "## Layout",
        "",
        "- `manifest.json` — list of all problems with filenames + status.",
        "- `images/` — cropped board images (PNG).",
        "- `metadata/` — per-problem JSON (stones at 19×19 col/row + status + source metadata).",
        "- `sgf/` — SGF with `SZ[19]`, `AB`/`AW` setup stones, `IM[]` reference to the image.",
        "",
        "## Status semantics",
        "",
        "- `accepted` — user reviewed and the detector's stone positions were correct.",
        "- `accepted_edited` — user reviewed and edited stone positions before accepting; ground truth is the edited version.",
        "- `rejected` — user rejected the detection outright. Stone positions are the detector's output (not ground truth); use for failure-mode analysis, not training signal.",
        "- `unreviewed` — no user decision.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source PDF file name")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--prefix", required=True, help="filename prefix (e.g. hm2)")
    args = ap.parse_args()
    manifest = export(args.source, args.out.expanduser(), args.prefix)
    print(f"exported {manifest['total']} problems → {args.out}")
    for k, v in manifest["counts"].items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
