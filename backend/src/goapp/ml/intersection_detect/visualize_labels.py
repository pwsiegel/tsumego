"""Render a few synth board crops with their on-the-fly intersection labels
drawn as green dots, so we can eyeball whether the labels actually land on
intersection centers.

Usage:
    uv --directory backend run python -m goapp.ml.intersection_detect.visualize_labels \\
        --n 8 --out /tmp/intersection_labels
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from ...paths import SYNTH_PAGES_DIR
from .train import _visible_intersections


def render_one(
    jp: Path, board_idx: int, out_path: Path,
    include_edges: bool = True,
) -> bool:
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

    # Draw the tight bbox as a faint blue rectangle so we can see whether
    # labels line up with grid extent.
    tx0, ty0, tx1, ty1 = board["bbox"]
    cv2.rectangle(
        crop,
        (int(tx0) - bx0, int(ty0) - by0),
        (int(tx1) - bx0, int(ty1) - by0),
        (255, 120, 0), 1,
    )

    # Draw labeled intersections.
    for ix, iy in _visible_intersections(board, include_edges=include_edges):
        cx = int(round(ix - bx0))
        cy = int(round(iy - by0))
        if 0 <= cx < crop.shape[1] and 0 <= cy < crop.shape[0]:
            cv2.circle(crop, (cx, cy), 3, (0, 200, 0), -1)
            cv2.circle(crop, (cx, cy), 3, (0, 90, 0), 1)

    cv2.imwrite(str(out_path), crop)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=SYNTH_PAGES_DIR)
    ap.add_argument("--out", type=Path, default=Path("/tmp/intersection_labels"))
    ap.add_argument("--n", type=int, default=12, help="number of crops to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-edge-labels", action="store_true",
                    help="Drop labels at actual go-board edges (matches the "
                         "training flag of the same name).")
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
        if render_one(jp, bi, out_path, include_edges=not args.no_edge_labels):
            print(f"wrote {out_path}")
            written += 1

    print(f"\n{written} crops → {args.out}")


if __name__ == "__main__":
    main()
