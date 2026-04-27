"""Intersection detection via the trained YOLOv8 detector.

One class: 0 = X (visible non-occluded "+"). Each prediction is a tiny
bbox around an intersection; the bbox center is the intersection point.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from ...paths import INTERSECTION_DETECTOR_PATH as DEFAULT_MODEL_PATH  # noqa: E402

PEAK_THRESH = 0.3
TRAIN_IMG_SIZE = 640

# NMS IoU. Intersections are 1 pitch apart; their bboxes (~0.5·pitch wide)
# don't overlap, so a low threshold is fine and avoids accidental merging.
INTERSECTION_NMS_IOU = 0.3


class IntersectionModelNotLoaded(RuntimeError):
    pass


def model_available(model_path: Path = DEFAULT_MODEL_PATH) -> bool:
    return model_path.exists()


@lru_cache(maxsize=4)
def _load_model(model_path: Path):
    if not model_path.exists():
        raise IntersectionModelNotLoaded(f"model file not found: {model_path}")
    from ultralytics import YOLO
    log.info("loading intersection YOLO from %s", model_path)
    return YOLO(str(model_path))


def detect_intersections_cnn(
    crop_bgr: np.ndarray,
    peak_thresh: float = PEAK_THRESH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[dict]:
    """Run YOLO on a board crop; return detected visible intersection centers.

    Each entry: {"x", "y", "conf"} in the crop's pixel coordinate space.
    """
    model = _load_model(model_path)
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    results = model.predict(
        crop_bgr,
        imgsz=TRAIN_IMG_SIZE,
        conf=float(peak_thresh),
        iou=INTERSECTION_NMS_IOU,
        # TTA is on for stones (sparse, robust). For intersections (dense,
        # 1-pitch apart) the multi-scale duplicates aren't always merged
        # cleanly and survive as offset ghosts — turn it off.
        augment=False,
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

    # Dedupe near-duplicates: anything within ~quarter-pitch of an already-
    # kept detection is a duplicate. We don't know the pitch directly, but
    # each predicted box is ~0.5·pitch wide, so median(box_dim)/2 ≈
    # 0.25·pitch — well below the 1·pitch spacing of true neighbors.
    detections.sort(key=lambda d: -d["conf"])
    half_proxy = float(max(1.0, np.median([
        max(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in xyxy
    ]) / 2.0))
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
