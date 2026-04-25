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
    """Mirror main._discretize_board on a raw crop. Returns a set of
    (col, row, color) tuples — the final discretized stones."""
    from ..ml.discretize.discretize import discretize
    from ..ml.pipeline import _resolve_geometry
    from ..ml.stone_detect.detect import detect_stones_cnn

    h, w = crop_bgr.shape[:2]
    stones = detect_stones_cnn(crop_bgr, peak_thresh=peak_thresh)

    pitch_x, pitch_y, ox, oy, edges = _resolve_geometry(crop_bgr)

    if pitch_x and pitch_y and pitch_x > 0 and pitch_y > 0:
        BOARD_MAX = 18
        top_b = (oy - pitch_y * 0.5) if oy is not None else -1e9
        bot_b = min(h, oy + BOARD_MAX * pitch_y + pitch_y * 0.5) if oy is not None else 1e9
        left_b = (ox - pitch_x * 0.5) if ox is not None else -1e9
        right_b = min(w, ox + BOARD_MAX * pitch_x + pitch_x * 0.5) if ox is not None else 1e9
        stones = [s for s in stones
                  if top_b <= s["y"] <= bot_b and left_b <= s["x"] <= right_b]

        HOSHI = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                 (15, 3), (15, 9), (15, 15)}
        if ox is not None and oy is not None:
            vc = max(1, min(19, int((w - 1 - ox) / pitch_x) + 1))
            vr = max(1, min(19, int((h - 1 - oy) / pitch_y) + 1))
            cmin = 0 if edges.get("left") else (19 - vc if edges.get("right") else max(0, (19 - vc) // 2))
            rmin = 0 if edges.get("top") else (19 - vr if edges.get("bottom") else max(0, (19 - vr) // 2))
            hoshi_r = 0.28 * min(pitch_x, pitch_y)
            def _not_hoshi(s):
                cl = max(0, min(vc - 1, int(round((s["x"] - ox) / pitch_x))))
                rl = max(0, min(vr - 1, int(round((s["y"] - oy) / pitch_y))))
                if (cmin + cl, rmin + rl) in HOSHI and s.get("r", 0) < hoshi_r:
                    return False
                return True
            stones = [s for s in stones if _not_hoshi(s)]

    pitch = (pitch_x + pitch_y) / 2 if pitch_x and pitch_y else pitch_x or pitch_y
    d = discretize(
        stones, w, h, edges=edges,
        cell_size_override=pitch,
        pitch_x_override=pitch_x, pitch_y_override=pitch_y,
        origin_x_override=ox, origin_y_override=oy,
    )
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
