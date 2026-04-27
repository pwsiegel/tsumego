"""Run a stone-detector model through the full discretize pipeline on
every problem in a validation dataset, and report diffs vs the saved
ground truth.

Built for the "did my retrain regress any accepted problems?" question.
Defaults filter to status=accepted (user-verified ground truth). For
each problem, runs: YOLO stones → edge detector → pitch → bounds/hoshi
filter → discretize, and compares discrete (col, row, color) against
what's in the val metadata.

Usage:
    uv --directory backend run python -m goapp.cli.compare_on_val \\
        --val-dir ~/data/go-app/data/val/hm2 \\
        --model ~/data/go-app/models/stone_detector_yolo.pt \\
        --status accepted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _run_pipeline(crop_bgr, peak_thresh: float):
    """Mirror the API's _discretize_board on a raw crop. Returns a set
    of (col, row, color) tuples — the final discretized stones."""
    from ..ml.pipeline import discretize_crop

    d, _edges = discretize_crop(crop_bgr, peak_thresh=peak_thresh)
    by_cell: dict[tuple[int, int], tuple[float, str]] = {}
    for s in d.stones:
        key = (s.col, s.row)
        prev = by_cell.get(key)
        if prev is None or s.conf > prev[0]:
            by_cell[key] = (s.conf, s.color)
    return {(c, r, col) for (c, r), (_, col) in by_cell.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True,
                    help="path to YOLO weights to test (overrides the live model)")
    ap.add_argument("--status", default="accepted",
                    help="only compare problems with this status (default: accepted)")
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    ap.add_argument("--verbose", action="store_true",
                    help="show per-diff detail for each changed problem")
    args = ap.parse_args()

    val_dir = args.val_dir.expanduser()
    model_path = args.model.expanduser()

    # Point the cached model loader at the requested weights without
    # touching the live install. Must be set BEFORE any detect_stones_cnn
    # call so the lru_cache picks up the override.
    from ..ml.stone_detect import detect as si
    si.MODEL_PATH = model_path
    si._load_model.cache_clear()
    print(f"using model: {model_path}")

    manifest = json.loads((val_dir / "manifest.json").read_text())
    entries = [p for p in manifest["problems"] if p["status"] == args.status]
    if not entries:
        raise SystemExit(f"no problems with status={args.status!r} in {val_dir}")
    print(f"comparing {len(entries)} problems (status={args.status})")

    unchanged = 0
    changed: list[dict] = []
    errors: list[dict] = []

    for i, entry in enumerate(entries):
        stem = entry["stem"]
        meta = json.loads((val_dir / "metadata" / f"{stem}.json").read_text())
        gt = {(s["col"], s["row"], s["color"]) for s in meta.get("stones", [])}

        img_path = val_dir / "images" / f"{stem}.png"
        crop = cv2.imread(str(img_path))
        if crop is None:
            errors.append({"stem": stem, "error": "could not read image"})
            continue

        try:
            pred = _run_pipeline(crop, args.peak_thresh)
        except Exception as e:
            errors.append({"stem": stem, "error": f"pipeline failed: {e}"})
            continue

        if pred == gt:
            unchanged += 1
        else:
            missed = gt - pred                        # ground truth not predicted
            extra = pred - gt                         # predicted not in ground truth
            # Color flips (same position, different color)
            gt_pos = {(c, r): col for (c, r, col) in gt}
            pred_pos = {(c, r): col for (c, r, col) in pred}
            flips = [
                (c, r, gt_pos[(c, r)], pred_pos[(c, r)])
                for (c, r) in gt_pos.keys() & pred_pos.keys()
                if gt_pos[(c, r)] != pred_pos[(c, r)]
            ]
            # Remove flip positions from missed/extra to avoid double-counting.
            flip_positions = {(c, r) for (c, r, _, _) in flips}
            missed = {(c, r, col) for (c, r, col) in missed if (c, r) not in flip_positions}
            extra = {(c, r, col) for (c, r, col) in extra if (c, r) not in flip_positions}
            changed.append({
                "stem": stem, "source_board_idx": entry["source_board_idx"],
                "gt_count": len(gt), "pred_count": len(pred),
                "missed": sorted(missed),
                "extra": sorted(extra),
                "flips": flips,
            })

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(entries)}]  unchanged={unchanged} changed={len(changed)} errors={len(errors)}")

    print()
    print(f"=== results on {len(entries)} {args.status} problems ===")
    print(f"  unchanged: {unchanged} ({unchanged / len(entries) * 100:.1f}%)")
    print(f"  changed:   {len(changed)} ({len(changed) / len(entries) * 100:.1f}%)")
    print(f"  errors:    {len(errors)}")

    if changed:
        missed_total = sum(len(c["missed"]) for c in changed)
        extra_total = sum(len(c["extra"]) for c in changed)
        flip_total = sum(len(c["flips"]) for c in changed)
        print()
        print(f"  across the {len(changed)} changed problems:")
        print(f"    stones newly MISSING : {missed_total}")
        print(f"    stones newly ADDED   : {extra_total}")
        print(f"    stones with COLOR FLIP: {flip_total}")

    if args.verbose and changed:
        print()
        print("--- per-problem diffs ---")
        for c in changed:
            print(f"  {c['stem']} (board #{c['source_board_idx']})  "
                  f"gt={c['gt_count']} pred={c['pred_count']}")
            if c["missed"]:
                print(f"    MISSED: {c['missed']}")
            if c["extra"]:
                print(f"    EXTRA : {c['extra']}")
            if c["flips"]:
                print(f"    FLIPS : {c['flips']}")

    if errors:
        print()
        print("--- errors ---")
        for e in errors:
            print(f"  {e['stem']}: {e['error']}")


if __name__ == "__main__":
    main()
