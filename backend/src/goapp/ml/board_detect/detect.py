"""Board detection via the trained YOLOv8 detector (ONNX runtime)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ...paths import BOARD_DETECTOR_ONNX as MODEL_PATH
from .. import _yolo_onnx

log = logging.getLogger(__name__)

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


def detect_boards_yolo(image_bgr: np.ndarray) -> list[BoardBBox]:
    """Run YOLO inference; return detected boards as BoardBBox list."""
    if not MODEL_PATH.exists():
        raise ModelNotLoaded(f"model file not found: {MODEL_PATH}")
    dets = _yolo_onnx.predict(
        MODEL_PATH, image_bgr,
        conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD,
    )

    out: list[BoardBBox] = []
    for d in dets:
        w = max(1.0, d.x1 - d.x0)
        h = max(1.0, d.y1 - d.y0)
        aspect = h / w
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            log.info("rejecting bbox with aspect %.2f", aspect)
            continue
        out.append(BoardBBox(
            x0=int(d.x0), y0=int(d.y0), x1=int(d.x1), y1=int(d.y1),
            confidence=d.conf,
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


def warm() -> None:
    """Pre-load the ONNX session and run one dummy inference."""
    if not MODEL_PATH.exists():
        raise ModelNotLoaded(f"model file not found: {MODEL_PATH}")
    _yolo_onnx.predict(
        MODEL_PATH, np.zeros((640, 640, 3), dtype=np.uint8),
        conf_thresh=0.99, iou_thresh=0.99,
    )
