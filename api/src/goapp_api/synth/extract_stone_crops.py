"""Turn synth_pages/ into stone-detector training crops.

For each synthetic page + its JSON annotation, crop every annotated board
bbox and emit:
    <crop_id>.png   — the cropped board image
    <crop_id>.json  — {"black": [[x, y], ...], "white": [[x, y], ...]}
in a stone-point-labels format matching the hand-labeled stone_points/ dir.

Usage:
    uv --directory api run python -m goapp_api.synth.extract_stone_crops
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import cv2
import numpy as np

from ..paths import (
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
    SYNTH_STONE_CROPS_DIR as DEFAULT_OUT,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    page_jsons = sorted(args.pages.glob("*.json"))
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
            x0, y0, x1, y1 = board["bbox"]
            x0 = max(0, int(x0)); y0 = max(0, int(y0))
            x1 = min(W, int(x1) + 1); y1 = min(H, int(y1) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = page[y0:y1, x0:x1]
            blacks = [[sx - x0, sy - y0] for (sx, sy, c) in board.get("stones", []) if c == "B"]
            whites = [[sx - x0, sy - y0] for (sx, sy, c) in board.get("stones", []) if c == "W"]
            # Keep only stones that end up inside the crop (rotations can
            # push a stone slightly outside its own bbox).
            cw, ch = crop.shape[1], crop.shape[0]
            blacks = [[x, y] for x, y in blacks if 0 <= x < cw and 0 <= y < ch]
            whites = [[x, y] for x, y in whites if 0 <= x < cw and 0 <= y < ch]
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            crop_id = f"synth_{stamp}_{secrets.token_hex(4)}"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            (out / f"{crop_id}.png").write_bytes(buf.tobytes())
            (out / f"{crop_id}.json").write_text(json.dumps({
                "task_id": crop_id,
                "black": blacks, "white": whites,
            }))
            count += 1

    print(f"wrote {count} crops to {out}")


if __name__ == "__main__":
    main()
