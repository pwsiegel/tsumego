"""Board detection via a fine-tuned YOLOv8 model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from ...paths import BOARD_DETECTOR_PATH as MODEL_PATH  # noqa: E402
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
# Real Go boards (even a narrow top/bottom strip of 19×5) cap around 4:1.
# Anything wider than that is almost certainly a caption / title banner.
MAX_ASPECT_RATIO = 4.0
MIN_ASPECT_RATIO = 1.0 / MAX_ASPECT_RATIO


@dataclass(frozen=True)
class BoardBBox:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


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
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        aspect = h / w
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            log.info("rejecting bbox with aspect %.2f", aspect)
            continue
        out.append(BoardBBox(
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            confidence=float(conf),
        ))
    # Reading order: top-to-bottom within each column, columns left-to-right.
    # Problems in Go books are typically laid out as a grid; a board's
    # "column" is determined by bucketing its x-center against the typical
    # board width so minor horizontal jitter doesn't split a column.
    if out:
        widths = sorted(b.x1 - b.x0 for b in out)
        median_w = max(1, widths[len(widths) // 2])
        bucket = max(1, median_w)
        def reading_key(b: BoardBBox) -> tuple[int, int]:
            cx = (b.x0 + b.x1) // 2
            cy = (b.y0 + b.y1) // 2
            return (cx // bucket, cy)
        out.sort(key=reading_key)
    log.info("detect_boards_yolo: %d boards", len(out))
    return out


def model_available() -> bool:
    return MODEL_PATH.exists()
