"""Fine-tune YOLOv8-nano on synthetic Go-stone annotations.

For each synth page (under $GOAPP_DATA_DIR/data/synth_pages/) and every
stone in its JSON, emit a YOLO bbox: class 0 for black, class 1 for white,
and a square bbox centered on the stone with side ≈ 0.8 × pitch. Train
YOLO end-to-end and save the best weights to
$GOAPP_DATA_DIR/models/stone_detector_yolo.pt (staging — we swap it in
as stone_detector.pt once the UNet path is rewired).

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_stones_yolo
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


from .paths import (
    DATA_DIR,
    MODELS_DIR,
    MODELS_RUNS_DIR,
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
)

SEED = 42
VAL_FRAC = 0.2
DEFAULT_EPOCHS = 20
IMG_SIZE = 640
STONE_BOX_FRAC = 0.4  # bbox half-side as a fraction of pitch

YOLO_DIR = DATA_DIR / "yolo_stones"
STAGED_OUT = MODELS_DIR / "stone_detector_yolo.pt"


def _pitch_from_board(board: dict) -> tuple[float, float]:
    """Recover (pitch_x_px, pitch_y_px) from a synth board annotation."""
    x0, y0, x1, y1 = board["bbox"]
    col_min, col_max, row_min, row_max = board["window"]
    n_cols = max(1, col_max - col_min)
    n_rows = max(1, row_max - row_min)
    return (x1 - x0) / n_cols, (y1 - y0) / n_rows


def build_yolo_dataset(pages_dir: Path, limit: int | None) -> Path:
    labels = sorted(pages_dir.glob("*.json"))
    if limit:
        labels = labels[:limit]
    if not labels:
        raise RuntimeError(f"no page JSONs in {pages_dir}")

    rng = random.Random(SEED)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - VAL_FRAC)))
    train_labels, val_labels = shuffled[:split_at], shuffled[split_at:]

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True)

    def emit(label_path: Path, split: str) -> None:
        img_path = label_path.with_suffix(".png")
        if not img_path.exists():
            return
        data = json.loads(label_path.read_text())
        with Image.open(img_path) as im:
            W, H = im.size

        stem = label_path.stem
        (YOLO_DIR / f"images/{split}" / f"{stem}.png").write_bytes(img_path.read_bytes())

        lines: list[str] = []
        for board in data.get("boards", []):
            px, py = _pitch_from_board(board)
            half = max(px, py) * STONE_BOX_FRAC
            for sx, sy, color in board.get("stones", []):
                cx, cy = float(sx), float(sy)
                cls = 0 if color == "B" else 1
                bw = 2 * half
                bh = 2 * half
                lines.append(
                    f"{cls} {cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f}"
                )
        (YOLO_DIR / f"labels/{split}" / f"{stem}.txt").write_text("\n".join(lines))

    for lp in train_labels:
        emit(lp, "train")
    for lp in val_labels:
        emit(lp, "val")

    yaml_text = (
        f"path: {YOLO_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: B\n"
        "  1: W\n"
    )
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"dataset: {len(train_labels)} train / {len(val_labels)} val → {YOLO_DIR}")
    return yaml_path


def train(data_yaml: Path, epochs: int, model_out: Path, base_model: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        project=str(MODELS_RUNS_DIR),
        name="stone_detector",
        exist_ok=True,
        patience=4,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, model_out)
    print(f"saved best weights → {model_out}")
    return model_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--limit", type=int, default=1200,
                    help="number of synth pages to include in training set")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--model-out", type=Path, default=STAGED_OUT)
    ap.add_argument("--base-model", type=str, default="yolov8n.pt")
    args = ap.parse_args()

    yaml_path = build_yolo_dataset(args.pages, args.limit)
    train(yaml_path, args.epochs, args.model_out, args.base_model)


if __name__ == "__main__":
    main()
