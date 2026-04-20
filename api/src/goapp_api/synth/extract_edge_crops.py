"""Turn synth_pages/ into per-bbox edge-classifier training crops.

For each page JSON, crop each board bbox out of the page PNG and emit:
    <crop_id>.png     — the cropped board image
    <crop_id>.json    — {"left": bool, "right": bool, "top": bool, "bottom": bool}

Usage:
    uv --directory api run python -m goapp_api.synth.extract_edge_crops
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
    SYNTH_EDGE_CROPS_DIR as DEFAULT_OUT,
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after processing this many pages (for quick preview)")
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
            x0, y0, x1, y1 = board["bbox"]
            x0 = max(0, int(x0)); y0 = max(0, int(y0))
            x1 = min(W, int(x1) + 1); y1 = min(H, int(y1) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = page[y0:y1, x0:x1]
            edges = board.get("edges_on_board", {})
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            crop_id = f"edge_{stamp}_{secrets.token_hex(4)}"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            (out / f"{crop_id}.png").write_bytes(buf.tobytes())
            (out / f"{crop_id}.json").write_text(json.dumps({
                "left": bool(edges.get("left", False)),
                "right": bool(edges.get("right", False)),
                "top": bool(edges.get("top", False)),
                "bottom": bool(edges.get("bottom", False)),
            }))
            count += 1

    print(f"wrote {count} edge-labeled crops to {out}")


if __name__ == "__main__":
    main()
