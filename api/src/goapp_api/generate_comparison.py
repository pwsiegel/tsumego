"""Generate a JSON snapshot comparing two stone-detector models against
the ground truth in a hand-annotated val dataset.

Output JSON has one entry per problem where the old model's discretized
output differs from the new model's. Consumed by the frontend compare
page (web/src/Compare.tsx).

Usage:
    uv --directory api run python -m goapp_api.generate_comparison \\
        --val-dir ~/data/go-app/data/val/hm2 \\
        --old-model ~/data/go-app/models/stone_detector.pt \\
        --new-model ~/data/go-app/models/stone_detector_yolo.pt \\
        --out ~/data/go-app/data/val/hm2/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .compare_on_val import _run_pipeline


def _run_with_model(crop_bgr, model_path: Path, peak_thresh: float):
    """Clear the cached YOLO model, point it at `model_path`, run the
    full discretize pipeline, return a list of (col, row, color) tuples."""
    from . import stone_inference as si
    si.MODEL_PATH = model_path
    si._load_model.cache_clear()
    return _run_pipeline(crop_bgr, peak_thresh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--old-model", type=Path, required=True)
    ap.add_argument("--new-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status", default="accepted",
                    help='"all" includes every problem; comma-separated list '
                         'restricts to those statuses (e.g. '
                         '"accepted,accepted_edited")')
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    args = ap.parse_args()

    val_dir = args.val_dir.expanduser()
    old_model = args.old_model.expanduser()
    new_model = args.new_model.expanduser()

    manifest = json.loads((val_dir / "manifest.json").read_text())
    if args.status == "all":
        entries = list(manifest["problems"])
    else:
        allowed = {s.strip() for s in args.status.split(",")}
        entries = [p for p in manifest["problems"] if p["status"] in allowed]
    print(f"comparing {len(entries)} problems (status={args.status})")

    # Load every crop + ground truth once; we'll run each model in turn.
    inputs: list[dict] = []
    for e in entries:
        stem = e["stem"]
        meta = json.loads((val_dir / "metadata" / f"{stem}.json").read_text())
        crop = cv2.imread(str(val_dir / "images" / f"{stem}.png"))
        if crop is None:
            continue
        gt = [(s["col"], s["row"], s["color"]) for s in meta.get("stones", [])]
        h, w = crop.shape[:2]
        inputs.append({
            "stem": stem,
            "source_board_idx": e["source_board_idx"],
            "status": e.get("status", "unknown"),
            "crop": crop,
            "crop_w": w, "crop_h": h,
            "gt": gt,
        })

    # Run each model across the whole set so YOLO only loads once per model.
    print(f"running OLD model: {old_model}")
    for i, item in enumerate(inputs):
        if i % 20 == 0:
            print(f"  [{i}/{len(inputs)}]")
        try:
            pred = _run_with_model(item["crop"], old_model, args.peak_thresh)
        except Exception as e:
            pred = None
        item["old"] = sorted(pred) if pred is not None else None

    print(f"running NEW model: {new_model}")
    for i, item in enumerate(inputs):
        if i % 20 == 0:
            print(f"  [{i}/{len(inputs)}]")
        try:
            pred = _run_with_model(item["crop"], new_model, args.peak_thresh)
        except Exception as e:
            pred = None
        item["new"] = sorted(pred) if pred is not None else None

    # Keep only problems where OLD and NEW disagree.
    changed = []
    for item in inputs:
        if item["old"] is None or item["new"] is None:
            continue
        if set(item["old"]) == set(item["new"]):
            continue
        gt_set = {(c, r, col) for (c, r, col) in item["gt"]}
        old_set = set(item["old"])
        new_set = set(item["new"])
        changed.append({
            "stem": item["stem"],
            "source_board_idx": item["source_board_idx"],
            "crop_width": item["crop_w"],
            "crop_height": item["crop_h"],
            "status": item.get("status", "unknown"),
            "gt": [{"col": c, "row": r, "color": col} for (c, r, col) in item["gt"]],
            "old": [{"col": c, "row": r, "color": col} for (c, r, col) in sorted(old_set)],
            "new": [{"col": c, "row": r, "color": col} for (c, r, col) in sorted(new_set)],
            "old_matches_gt": old_set == gt_set,
            "new_matches_gt": new_set == gt_set,
        })

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "val_dir": str(val_dir),
        "old_model": str(old_model),
        "new_model": str(new_model),
        "status_filter": args.status,
        "total": len(inputs),
        "changed_count": len(changed),
        "problems": changed,
    }, indent=2))

    print()
    print(f"=== {len(changed)} problems where old ≠ new (of {len(inputs)} total) ===")
    print(f"  old matches ground truth: {sum(1 for c in changed if c['old_matches_gt'])}")
    print(f"  new matches ground truth: {sum(1 for c in changed if c['new_matches_gt'])}")
    print(f"  neither matches ground truth: {sum(1 for c in changed if not c['old_matches_gt'] and not c['new_matches_gt'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
