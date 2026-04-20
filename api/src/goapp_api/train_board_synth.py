"""Fine-tune YOLOv8-nano on the synthetic page dataset.

Reads `training_data/synth_pages/*.{png,json}`, converts to YOLO format,
splits train/val 80/20, trains, and saves the best weights to
`models/board_detector.pt`.

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_board_synth
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAGES = ROOT / "training_data" / "synth_pages"
YOLO_DIR = ROOT / "training_data" / "yolo_synth"
MODELS_DIR = ROOT / "models"
SEED = 42
VAL_FRAC = 0.2
DEFAULT_EPOCHS = 25
IMG_SIZE = 640


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
            w, h = im.size

        stem = label_path.stem
        (YOLO_DIR / f"images/{split}" / f"{stem}.png").write_bytes(img_path.read_bytes())

        lines = []
        for board in data.get("boards", []):
            # Tight bbox, single "board" class. Edge bits come from a
            # separate specialist classifier, not YOLO's class head.
            x0, y0, x1, y1 = board["bbox"]
            cx = ((x0 + x1) / 2) / w
            cy = ((y0 + y1) / 2) / h
            bw = (x1 - x0 + 1) / w
            bh = (y1 - y0 + 1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
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
        "  0: board\n"
    )
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"dataset: {len(train_labels)} train / {len(val_labels)} val → {YOLO_DIR}")
    return yaml_path


def train(data_yaml: Path, epochs: int, model_out: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        project=str(MODELS_DIR / "runs"),
        name="board_detector_synth",
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
    ap.add_argument("--limit", type=int, default=500,
                    help="number of synth pages to include in training set")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--model-out", type=Path, default=MODELS_DIR / "board_detector.pt")
    args = ap.parse_args()

    yaml_path = build_yolo_dataset(args.pages, args.limit)
    train(yaml_path, args.epochs, args.model_out)


if __name__ == "__main__":
    main()
