"""Fine-tune YOLOv8-nano to detect visible perpendicular grid intersections.

One class:

    0: X    a non-occluded "+"-shaped intersection on a Go board grid

Labels are computed on the fly from each board's tight bbox + window:
the lattice is implicit in the synth annotation (tight bbox aligns with
the outermost intersections), so for every (col, row) inside the window
we drop a tiny bbox at the projected page coords and skip any whose
center sits within ~half a pitch of a stone center (those are occluded).
Hoshi star points are kept — they ARE intersections, just with a small
ornament that the detector should still see as a "+".

At inference, lattice fitting combines the detected intersections with
the stone detector's centers to recover the full grid (covered + not).

Usage:
    uv --directory backend run --extra ml python -m goapp.ml.intersection_detect.train \\
        --device mps     # use Apple M-series GPU if available
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


from ...paths import (
    DATA_DIR,
    MODELS_DIR,
    TRAINING_RUNS_DIR,
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
)

SEED = 42
VAL_FRAC = 0.2
DEFAULT_EPOCHS = 30
IMG_SIZE = 640

# Bbox half-size as a fraction of pitch. Smaller than stones (0.4) since a
# "+" is a more localized feature — a tight box helps NMS keep adjacent
# intersections (1 pitch apart) from being merged.
INTERSECTION_HALF_FRAC = 0.25

# An intersection is "occluded" iff a stone center lies within this many
# pitch-units of it. Stones span ~0.9·pitch, so 0.5·pitch is a clean cut.
OCCLUSION_PITCH_FRAC = 0.5

YOLO_DIR = DATA_DIR / "yolo_intersections"
STAGED_OUT = MODELS_DIR / "intersection_detector.pt"

CLASS_NAMES = ["X"]


def _pitch_from_board(board: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = board["bbox"]
    col_min, col_max, row_min, row_max = board["window"]
    n_cols = max(1, col_max - col_min)
    n_rows = max(1, row_max - row_min)
    return (x1 - x0) / n_cols, (y1 - y0) / n_rows


def _visible_intersections(board: dict) -> list[tuple[float, float]]:
    """Return page-pixel centers of every visible non-occluded intersection."""
    x0, y0, x1, y1 = board["bbox"]
    col_min, col_max, row_min, row_max = board["window"]
    px, py = _pitch_from_board(board)
    occlusion_r2 = (OCCLUSION_PITCH_FRAC * max(px, py)) ** 2
    stones = board.get("stones", [])

    out: list[tuple[float, float]] = []
    for r_idx in range(row_min, row_max + 1):
        for c_idx in range(col_min, col_max + 1):
            cx = x0 + (c_idx - col_min) * px
            cy = y0 + (r_idx - row_min) * py
            occluded = False
            for sx, sy, _color in stones:
                if (sx - cx) ** 2 + (sy - cy) ** 2 < occlusion_r2:
                    occluded = True
                    break
            if not occluded:
                out.append((cx, cy))
    return out


def _emit_box(
    lines: list[str],
    cls: int,
    cx: float, cy: float,
    half: float,
    W: int, H: int,
) -> None:
    lines.append(
        f"{cls} {cx / W:.6f} {cy / H:.6f} {2 * half / W:.6f} {2 * half / H:.6f}"
    )


def build_yolo_dataset(pages_dir: Path, limit: int | None) -> Path:
    """Extract per-board crops and emit a YOLO dataset of intersection bboxes."""
    page_jsons = sorted(pages_dir.glob("*.json"))
    if limit:
        page_jsons = page_jsons[:limit]
    if not page_jsons:
        raise RuntimeError(f"no page JSONs in {pages_dir}")

    pairs: list[tuple[Path, int]] = []
    for jp in page_jsons:
        data = json.loads(jp.read_text())
        for i in range(len(data.get("boards", []))):
            pairs.append((jp, i))
    if not pairs:
        raise RuntimeError("no boards in any page JSON")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    split_at = max(1, int(len(pairs) * (1 - VAL_FRAC)))
    train_pairs, val_pairs = pairs[:split_at], pairs[split_at:]

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True)

    _page_cache: dict[Path, np.ndarray] = {}

    def decode(png_path: Path) -> np.ndarray | None:
        if png_path not in _page_cache:
            if not png_path.exists():
                return None
            arr = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
            _page_cache[png_path] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return _page_cache[png_path]

    def emit(jp: Path, board_idx: int, split: str) -> None:
        data = json.loads(jp.read_text())
        board = data["boards"][board_idx]
        page = decode(jp.with_suffix(".png"))
        if page is None:
            return
        Hp, Wp = page.shape[:2]
        bx0, by0, bx1, by1 = board.get("bbox_padded", board["bbox"])
        bx0 = max(0, int(bx0)); by0 = max(0, int(by0))
        bx1 = min(Wp, int(bx1) + 1); by1 = min(Hp, int(by1) + 1)
        if bx1 <= bx0 or by1 <= by0:
            return
        crop = page[by0:by1, bx0:bx1]
        Ch, Cw = crop.shape[:2]

        px, py = _pitch_from_board(board)
        half = max(px, py) * INTERSECTION_HALF_FRAC

        lines: list[str] = []
        for ix, iy in _visible_intersections(board):
            cx = ix - bx0
            cy = iy - by0
            if not (0 <= cx < Cw and 0 <= cy < Ch):
                continue
            _emit_box(lines, 0, float(cx), float(cy), half, Cw, Ch)

        stem = f"{jp.stem}_b{board_idx}"
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return
        (YOLO_DIR / f"images/{split}" / f"{stem}.png").write_bytes(buf.tobytes())
        (YOLO_DIR / f"labels/{split}" / f"{stem}.txt").write_text("\n".join(lines))

    for jp, bi in train_pairs:
        emit(jp, bi, "train")
    for jp, bi in val_pairs:
        emit(jp, bi, "val")

    yaml_text = (
        f"path: {YOLO_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"dataset: {len(train_pairs)} train / {len(val_pairs)} val crops → {YOLO_DIR}")
    return yaml_path


def train(data_yaml: Path, epochs: int, model_out: Path, base_model: str,
          device: str | None = None) -> Path:
    from ultralytics import YOLO
    model = YOLO(base_model)
    train_kwargs: dict = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        project=str(TRAINING_RUNS_DIR),
        name="intersection_detector",
        exist_ok=True,
        patience=4,
        # Intersections look the same under all of these — augment freely.
        degrees=3.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.3,
    )
    if device:
        train_kwargs["device"] = device
        if device == "mps":
            train_kwargs["amp"] = False  # MPS + AMP causes tensor shape mismatches
    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, model_out)
    print(f"saved best weights → {model_out}")
    return model_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--limit", type=int, default=1500,
                    help="number of synth pages to process")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--model-out", type=Path, default=STAGED_OUT)
    ap.add_argument("--base-model", type=str, default="yolov8n.pt")
    ap.add_argument("--device", type=str, default=None,
                    help='Training device: "cpu", "mps" (Apple M-series GPU), '
                         '"0" (first CUDA GPU), etc. Default lets Ultralytics choose.')
    args = ap.parse_args()

    yaml_path = build_yolo_dataset(args.pages, args.limit)
    train(yaml_path, args.epochs, args.model_out, args.base_model, args.device)


if __name__ == "__main__":
    main()
