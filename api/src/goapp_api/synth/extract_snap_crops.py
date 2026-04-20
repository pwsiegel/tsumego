"""Turn synth_pages/ into snap-to-grid training examples.

For each page's boards, jitter the tight bbox by a random amount on each
side (simulates YOLO's slack at inference), crop the jittered region out of
the page, and compute the ground-truth (pitch_x_frac, pitch_y_frac,
origin_x_frac, origin_y_frac) for that jittered crop.

Emits:
    <id>.png   — the jittered crop
    <id>.json  — {"pitch_x": float, "pitch_y": float,
                   "origin_x": float, "origin_y": float}

All four values are fractions of crop width/height, so they stay scale
invariant when the image is resized to IMG_SIZE for training.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from pathlib import Path

import cv2
import numpy as np


from ..paths import (
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
    SYNTH_SNAP_CROPS_DIR as DEFAULT_OUT,
)
SEED = 42
JITTER_MAX_FRAC = 0.10   # up to 10% pad/trim on each side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N pages")
    ap.add_argument("--per-board", type=int, default=3,
                    help="number of random jitter samples per board")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    page_jsons = sorted(args.pages.glob("*.json"))
    if args.limit:
        page_jsons = page_jsons[:args.limit]

    count = 0
    for json_path in page_jsons:
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            continue
        data = json.loads(json_path.read_text())
        page = cv2.imdecode(
            np.frombuffer(png_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if page is None:
            continue
        H, W = page.shape[:2]

        for board in data.get("boards", []):
            # Tight bbox + window.
            bx0, by0, bx1, by1 = (int(v) for v in board["bbox"])
            window = board.get("window")
            if window is None:
                continue
            col_min, col_max, row_min, row_max = (int(v) for v in window)
            if col_max <= col_min or row_max <= row_min:
                continue
            orig_w = bx1 - bx0
            orig_h = by1 - by0
            if orig_w <= 0 or orig_h <= 0:
                continue
            # Pitch in ORIGINAL page coordinates.
            pitch_page_x = orig_w / (col_max - col_min)
            pitch_page_y = orig_h / (row_max - row_min)

            for _ in range(args.per_board):
                # Random jitter: positive = pad outward, negative = trim inward.
                jl = int(rng.uniform(-JITTER_MAX_FRAC, JITTER_MAX_FRAC) * orig_w)
                jr = int(rng.uniform(-JITTER_MAX_FRAC, JITTER_MAX_FRAC) * orig_w)
                jt = int(rng.uniform(-JITTER_MAX_FRAC, JITTER_MAX_FRAC) * orig_h)
                jb = int(rng.uniform(-JITTER_MAX_FRAC, JITTER_MAX_FRAC) * orig_h)

                ex0, ey0 = bx0 - jl, by0 - jt
                ex1, ey1 = bx1 + jr, by1 + jb
                # Stay within the page.
                ex0 = max(0, ex0); ey0 = max(0, ey0)
                ex1 = min(W, ex1); ey1 = min(H, ey1)
                crop_w = ex1 - ex0
                crop_h = ey1 - ey0
                if crop_w <= 10 or crop_h <= 10:
                    continue
                crop = page[ey0:ey1, ex0:ex1]

                # Target: where is col_min in the crop? bx0 in page →
                # (bx0 - ex0) in crop.
                origin_x = (bx0 - ex0)  # pixels in crop
                origin_y = (by0 - ey0)
                # Pitch in crop space: same as pitch in page (the crop
                # didn't resize anything).
                pitch_x = pitch_page_x
                pitch_y = pitch_page_y
                # Fractions (scale-invariant).
                px_frac = pitch_x / crop_w
                py_frac = pitch_y / crop_h
                ox_frac = origin_x / crop_w
                oy_frac = origin_y / crop_h

                stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
                crop_id = f"snap_{stamp}_{secrets.token_hex(4)}"
                ok, buf = cv2.imencode(".png", crop)
                if not ok:
                    continue
                (args.out / f"{crop_id}.png").write_bytes(buf.tobytes())
                (args.out / f"{crop_id}.json").write_text(json.dumps({
                    "pitch_x": float(px_frac),
                    "pitch_y": float(py_frac),
                    "origin_x": float(ox_frac),
                    "origin_y": float(oy_frac),
                    "window": [col_min, col_max, row_min, row_max],
                }))
                count += 1
                if count % 1000 == 0:
                    print(f"  extracted {count}", flush=True)

    print(f"wrote {count} snap crops to {args.out}")


if __name__ == "__main__":
    main()
