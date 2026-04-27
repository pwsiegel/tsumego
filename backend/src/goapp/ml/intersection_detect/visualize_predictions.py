"""Run the trained intersection detector on raw synth bbox_padded crops
(no PDF re-render, no board-detector in the loop) and draw GT labels +
predictions side-by-side.

Green dots: ground-truth labels (from on-the-fly synth labels).
Red dots:   predictions (filled = above peak_thresh).

If GT and predictions disagree on the SAME crops the model was trained on,
the model itself is the bottleneck. If they agree here but the inference
UI looks bad, the problem is the board-detector → PDF re-render path.

Usage:
    uv --directory backend run --extra ml python -m goapp.ml.intersection_detect.visualize_predictions \\
        --n 12 --out /tmp/intersection_preds
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from ...paths import SYNTH_PAGES_DIR
from .detect import detect_intersections_cnn
from .train import _visible_intersections


def render_one(jp: Path, board_idx: int, out_path: Path, peak_thresh: float) -> bool:
    data = json.loads(jp.read_text())
    boards = data.get("boards", [])
    if board_idx >= len(boards):
        return False
    board = boards[board_idx]

    png = jp.with_suffix(".png")
    if not png.exists():
        return False
    page = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if page is None:
        return False
    Hp, Wp = page.shape[:2]

    bx0, by0, bx1, by1 = board.get("bbox_padded", board["bbox"])
    bx0 = max(0, int(bx0)); by0 = max(0, int(by0))
    bx1 = min(Wp, int(bx1) + 1); by1 = min(Hp, int(by1) + 1)
    if bx1 <= bx0 or by1 <= by0:
        return False
    crop = page[by0:by1, bx0:bx1].copy()

    # Tight bbox outline (orange).
    tx0, ty0, tx1, ty1 = board["bbox"]
    cv2.rectangle(
        crop,
        (int(tx0) - bx0, int(ty0) - by0),
        (int(tx1) - bx0, int(ty1) - by0),
        (255, 120, 0), 1,
    )

    # GT labels (green).
    for ix, iy in _visible_intersections(board):
        cx = int(round(ix - bx0))
        cy = int(round(iy - by0))
        if 0 <= cx < crop.shape[1] and 0 <= cy < crop.shape[0]:
            cv2.circle(crop, (cx, cy), 4, (0, 200, 0), -1)

    # Predictions (red).
    preds = detect_intersections_cnn(crop, peak_thresh=peak_thresh)
    for p in preds:
        cx = int(round(p["x"]))
        cy = int(round(p["y"]))
        cv2.circle(crop, (cx, cy), 3, (0, 0, 220), -1)

    cv2.imwrite(str(out_path), crop)
    n_gt = sum(1 for _ in _visible_intersections(board))
    print(f"  GT={n_gt}  preds={len(preds)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=SYNTH_PAGES_DIR)
    ap.add_argument("--out", type=Path, default=Path("/tmp/intersection_preds"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--peak-thresh", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    page_jsons = sorted(args.pages.glob("*.json"))
    if not page_jsons:
        raise SystemExit(f"no page JSONs in {args.pages}")

    pairs: list[tuple[Path, int]] = []
    for jp in page_jsons:
        data = json.loads(jp.read_text())
        for i in range(len(data.get("boards", []))):
            pairs.append((jp, i))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for jp, bi in pairs:
        if written >= args.n:
            break
        out_path = args.out / f"{jp.stem}_b{bi}.png"
        print(f"{out_path.name}:")
        if render_one(jp, bi, out_path, args.peak_thresh):
            written += 1

    print(f"\n{written} crops → {args.out}")


if __name__ == "__main__":
    main()
