"""Turn synth_pages/ into grid-classifier training crops.

For each page JSON, crop each board bbox out of the page PNG and emit:
    <crop_id>.png    — cropped board
    <crop_id>.json   — {"grid": 19x19 array of 0=empty/1=B/2=W,
                        "mask": 19x19 array of 1 where the cell is visible
                                in the crop's window, 0 otherwise}

Partial-board crops only supervise the visible window; cells outside the
window are masked out of the loss so the model isn't punished for guessing
at them.

Usage:
    uv --directory api run python -m goapp_api.synth.extract_grid_crops
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAGES = ROOT / "training_data" / "synth_pages"
DEFAULT_OUT = ROOT / "training_data" / "synth_grid_crops"

BOARD_SIZE = 19
EMPTY, BLACK, WHITE = 0, 1, 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    page_jsons = sorted(args.pages.glob("*.json"))
    if args.limit:
        page_jsons = page_jsons[:args.limit]

    count = 0
    for json_path in page_jsons:
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            continue
        data = json.loads(json_path.read_text())
        arr = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
        page = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if page is None:
            continue
        H, W = page.shape[:2]
        for board in data.get("boards", []):
            # Use the TIGHT bbox — edges lie exactly at the outer visible
            # grid intersections, so pixel ↔ board-cell mapping is exact.
            bx0, by0, bx1, by1 = board["bbox"]
            bx0 = max(0, int(bx0)); by0 = max(0, int(by0))
            bx1 = min(W, int(bx1) + 1); by1 = min(H, int(by1) + 1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            crop = page[by0:by1, bx0:bx1]
            grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.uint8)
            mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
            window = board.get("window", [0, BOARD_SIZE - 1, 0, BOARD_SIZE - 1])
            col_min, col_max, row_min, row_max = [int(v) for v in window]
            mask[row_min:row_max + 1, col_min:col_max + 1] = 1

            # With tight bboxes, col_min's pixel is at bx0, col_max at bx1.
            # Pitch = (bx1 - bx0) / (col_max - col_min) exactly.
            if col_max > col_min and row_max > row_min:
                board_w = bx1 - bx0
                board_h = by1 - by0
                pitch_x = board_w / max(1, (col_max - col_min))
                pitch_y = board_h / max(1, (row_max - row_min))
                for sx, sy, color in board.get("stones", []):
                    rx = sx - bx0
                    ry = sy - by0
                    col = col_min + int(round(rx / pitch_x))
                    row = row_min + int(round(ry / pitch_y))
                    col = max(col_min, min(col_max, col))
                    row = max(row_min, min(row_max, row))
                    grid[row, col] = BLACK if color == "B" else WHITE
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            crop_id = f"grid_{stamp}_{secrets.token_hex(4)}"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            (out / f"{crop_id}.png").write_bytes(buf.tobytes())
            (out / f"{crop_id}.json").write_text(json.dumps({
                "grid": grid.tolist(),
                "mask": mask.tolist(),
                "window": [col_min, col_max, row_min, row_max],
            }))
            count += 1
            if count % 500 == 0:
                print(f"  extracted {count}")

    print(f"wrote {count} grid-labeled crops to {out}")


if __name__ == "__main__":
    main()
