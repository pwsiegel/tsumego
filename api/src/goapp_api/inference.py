"""Board detection via a fine-tuned YOLOv8 model."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .detection import BoardBBox

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "board_detector.pt"
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.4


class ModelNotLoaded(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotLoaded(f"model file not found: {MODEL_PATH}")
    from ultralytics import YOLO  # imported lazily so base deps don't need it
    log.info("loading YOLO model from %s", MODEL_PATH)
    return YOLO(str(MODEL_PATH))


def detect_boards_yolo(image_bgr: np.ndarray) -> list[BoardBBox]:
    """Run YOLO inference; return detected boards as BoardBBox list."""
    model = _load_model()
    # ultralytics accepts numpy images directly (it handles BGR/RGB internally).
    results = model.predict(
        image_bgr,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    if not results:
        return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    out: list[BoardBBox] = []
    for (x0, y0, x1, y1), conf in zip(xyxy, confs):
        c = int(conf * 1000)
        out.append(BoardBBox(
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            h_lines=c, v_lines=c,
        ))
    # Largest, highest-confidence first.
    out.sort(key=lambda b: -((b.x1 - b.x0) * (b.y1 - b.y0) * max(1, b.h_lines)))
    log.info("detect_boards_yolo: %d boards", len(out))
    return out


def model_available() -> bool:
    return MODEL_PATH.exists()
