"""Board detection via a fine-tuned YOLOv8 model."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .detection import BoardBBox

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "board_detector.pt"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
# Real Go boards (even a narrow top/bottom strip of 19×5) cap around 4:1.
# Anything wider than that is almost certainly a caption / title banner.
MAX_ASPECT_RATIO = 4.0
MIN_ASPECT_RATIO = 1.0 / MAX_ASPECT_RATIO


def decode_edge_class(cls: int) -> dict[str, bool]:
    """Decode the 4-bit edge encoding (L<<3 | R<<2 | T<<1 | B) into a dict."""
    return {
        "left": bool((cls >> 3) & 1),
        "right": bool((cls >> 2) & 1),
        "top": bool((cls >> 1) & 1),
        "bottom": bool(cls & 1),
    }


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
        c = int(conf * 1000)
        out.append(BoardBBox(
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            h_lines=c, v_lines=c,
        ))
    # Largest, highest-confidence first.
    out.sort(key=lambda b: -((b.x1 - b.x0) * (b.y1 - b.y0) * max(1, b.h_lines)))
    log.info("detect_boards_yolo: %d boards", len(out))
    return out


def detect_boards_with_edges(
    image_bgr: np.ndarray,
) -> list[tuple[BoardBBox, dict[str, bool]]]:
    """Run YOLO and return each detection with its predicted edge bits.

    YOLO's class id is the 4-bit edge encoding (L<<3|R<<2|T<<1|B), so decoding
    is deterministic. Returns [(bbox, {left,right,top,bottom}), ...].
    """
    model = _load_model()
    results = model.predict(
        image_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False,
    )
    if not results:
        return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    out: list[tuple[BoardBBox, dict[str, bool]]] = []
    for (x0, y0, x1, y1), conf, cls in zip(xyxy, confs, classes):
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        aspect = h / w
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue
        c = int(conf * 1000)
        bb = BoardBBox(
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            h_lines=c, v_lines=c,
        )
        out.append((bb, decode_edge_class(int(cls))))
    out.sort(key=lambda t: -((t[0].x1 - t[0].x0) * (t[0].y1 - t[0].y0) * max(1, t[0].h_lines)))
    return out


def model_available() -> bool:
    return MODEL_PATH.exists()
