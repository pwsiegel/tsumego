"""Intersection detection via the trained YOLOv8 detector.

One class: 0 = X (visible non-occluded "+"). Each prediction is a tiny
bbox around an intersection; the bbox center is the intersection point.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

from ...paths import INTERSECTION_DETECTOR_PATH as MODEL_PATH  # noqa: E402

PEAK_THRESH = 0.3
TRAIN_IMG_SIZE = 640

# NMS IoU. Intersections are 1 pitch apart; their bboxes (~0.5·pitch wide)
# don't overlap, so a low threshold is fine and avoids accidental merging.
INTERSECTION_NMS_IOU = 0.3


class IntersectionModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise IntersectionModelNotLoaded(f"model file not found: {MODEL_PATH}")
    from ultralytics import YOLO
    log.info("loading intersection YOLO from %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    return model


def detect_intersections_cnn(
    crop_bgr: np.ndarray,
    peak_thresh: float = PEAK_THRESH,
) -> list[dict]:
    """Run YOLO on a board crop; return detected visible intersection centers.

    Each entry: {"x", "y", "conf"} in the crop's pixel coordinate space.
    """
    model = _load_model()
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    results = model.predict(
        crop_bgr,
        imgsz=TRAIN_IMG_SIZE,
        conf=float(peak_thresh),
        iou=INTERSECTION_NMS_IOU,
        augment=True,
        verbose=False,
    )
    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()

    detections: list[dict] = []
    for (x0, y0, x1, y1), p in zip(xyxy, conf):
        detections.append({
            "x": float((x0 + x1) / 2.0),
            "y": float((y0 + y1) / 2.0),
            "conf": float(p),
        })

    # Dedupe TTA duplicates: anything within ~quarter-pitch of an already-
    # kept detection is a duplicate. We don't know the pitch yet, so use
    # the bbox half-size as a proxy (each box is ~0.5·pitch).
    detections.sort(key=lambda d: -d["conf"])
    half_proxy = float(max(1.0, np.median([
        max(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in xyxy
    ]) / 4.0))
    kept: list[dict] = []
    for d in detections:
        dup = False
        for k in kept:
            if (d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 < half_proxy ** 2:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept
