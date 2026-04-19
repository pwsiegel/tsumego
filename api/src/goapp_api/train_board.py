"""Fine-tune YOLOv8-nano on the saved board-label dataset.

Reads `training_data/boards/*.{png,json}`, converts to YOLO format,
splits train/val 80/20, trains, and saves the best weights to
`models/board_detector.pt`.

Usage:
    uv run --extra ml python -m goapp_api.train_board
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BOARDS_DIR = ROOT / "training_data" / "boards"
YOLO_DIR = ROOT / "training_data" / "yolo"
MODELS_DIR = ROOT / "models"
SEED = 42
VAL_FRAC = 0.2
EPOCHS = 50
IMG_SIZE = 640


def build_yolo_dataset() -> Path:
    """Convert labels in BOARDS_DIR to YOLO format. Returns path to data.yaml."""
    labels = sorted(BOARDS_DIR.glob("*.json"))
    if not labels:
        raise RuntimeError(f"no labels in {BOARDS_DIR}")

    rng = random.Random(SEED)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - VAL_FRAC)))
    train_labels, val_labels = shuffled[:split_at], shuffled[split_at:]

    # Wipe and rebuild yolo output dir.
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True)

    def emit(label_path: Path, split: str) -> None:
        img_path = label_path.with_suffix(".png")
        if not img_path.exists():
            print(f"[skip] no image for {label_path.name}")
            return
        data = json.loads(label_path.read_text())
        bboxes = data["bboxes"]
        with Image.open(img_path) as im:
            w, h = im.size

        stem = label_path.stem
        (YOLO_DIR / f"images/{split}" / f"{stem}.png").write_bytes(img_path.read_bytes())

        lines = []
        for x0, y0, x1, y1 in bboxes:
            # YOLO format: class_id cx cy w h (normalized 0-1).
            cx = ((x0 + x1) / 2) / w
            cy = ((y0 + y1) / 2) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
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


def train(data_yaml: Path) -> Path:
    from ultralytics import YOLO  # import here so base deps don't need it

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")  # pretrained on COCO
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(MODELS_DIR / "runs"),
        name="board_detector",
        exist_ok=True,
        patience=10,
    )
    # Copy best weights to a stable path.
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    target = MODELS_DIR / "board_detector.pt"
    shutil.copy(best, target)
    print(f"saved best weights → {target}")
    return target


def main() -> None:
    yaml_path = build_yolo_dataset()
    train(yaml_path)


if __name__ == "__main__":
    main()
