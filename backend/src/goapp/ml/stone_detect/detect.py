"""Stone detection via the trained YOLOv8 detector (ONNX runtime).

Two classes: 0 = B (black), 1 = W (white). Each prediction is a bbox
around a stone; we take the bbox center as the stone position.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ...paths import STONE_DETECTOR_ONNX as MODEL_PATH
from .. import _yolo_onnx

log = logging.getLogger(__name__)

PEAK_THRESH = 0.3  # kept as the `peak_thresh` kwarg name for API compat
TRAIN_IMG_SIZE = 640  # training imgsz; used as default for larger crops

# NMS IoU for stone detections. Stones are well-separated relative to their
# size, so a moderate threshold is fine.
STONE_NMS_IOU = 0.5

# Color reclassification samples a centered sub-square of the bbox. 1/3 of
# the radius keeps the sample well inside the stone, away from grid lines
# at the edges where the YOLO bbox can clip a printed line.
COLOR_SAMPLE_INNER_FRAC = 0.33

# Mean-gray cutoffs for forcing the color label after sampling. Below
# BLACK_GRAY_MAX → definitely black; above WHITE_GRAY_MIN → definitely
# white; in between we trust YOLO's class head.
BLACK_GRAY_MAX = 100
WHITE_GRAY_MIN = 180


class StoneModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


def warm() -> None:
    """Pre-load the ONNX session and run one dummy inference."""
    if not MODEL_PATH.exists():
        raise StoneModelNotLoaded(f"model file not found: {MODEL_PATH}")
    _yolo_onnx.predict(
        MODEL_PATH, np.zeros((TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, 3), dtype=np.uint8),
        conf_thresh=0.99, iou_thresh=0.99, imgsz=TRAIN_IMG_SIZE,
    )


def detect_stones_cnn(
    crop_bgr: np.ndarray,
    peak_thresh: float = PEAK_THRESH,
) -> list[dict]:
    """Run YOLO on a board crop; return detected stone centers.

    Each entry: {"x", "y", "r", "color", "conf"} in the crop's pixel
    coordinate space. `peak_thresh` is used as the YOLO confidence
    threshold (kept as kwarg name for API compatibility).
    """
    if not MODEL_PATH.exists():
        raise StoneModelNotLoaded(f"model file not found: {MODEL_PATH}")
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    dets = _yolo_onnx.predict(
        MODEL_PATH, crop_bgr,
        conf_thresh=float(peak_thresh), iou_thresh=STONE_NMS_IOU,
        imgsz=TRAIN_IMG_SIZE,
    )
    if not dets:
        return []

    # Post-classify color from the actual pixel values at each detection
    # center. Pixel darkness is unambiguous even when YOLO's class head
    # gets confused on lower-contrast scans.
    gray_img = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )

    detections: list[dict] = []
    for d in dets:
        cx = (d.x0 + d.x1) / 2.0
        cy = (d.y0 + d.y1) / 2.0
        r = max(d.x1 - d.x0, d.y1 - d.y0) / 2.0
        color = "B" if d.cls == 0 else "W"
        # Sample the center fraction of the bbox — avoids grid lines at
        # stone edges and captures the stone's actual fill color.
        inner = max(1, int(r * COLOR_SAMPLE_INNER_FRAC))
        ix0 = max(0, int(cx - inner))
        ix1 = min(gray_img.shape[1], int(cx + inner) + 1)
        iy0 = max(0, int(cy - inner))
        iy1 = min(gray_img.shape[0], int(cy + inner) + 1)
        if ix1 > ix0 and iy1 > iy0:
            mean_gray = float(gray_img[iy0:iy1, ix0:ix1].mean())
            if mean_gray < BLACK_GRAY_MAX:
                color = "B"
            elif mean_gray > WHITE_GRAY_MIN:
                color = "W"
        detections.append({
            "x": cx,
            "y": cy,
            "r": r,
            "color": color,
            "conf": d.conf,
        })

    # Deduplicate near-identical centers. (TTA used to produce these via
    # multi-scale + flip merging in the ultralytics path; the ONNX path
    # runs a single scale, but residual NMS overlap can still leave
    # duplicates within ~0.2·pitch of each other while adjacent-cell
    # stones are a full pitch apart. A threshold of half the smaller
    # radius (since r ≈ 0.4·pitch, this is ~0.2·pitch) keeps adjacent
    # stones separate while collapsing overlapping detections.
    detections.sort(key=lambda d: -d["conf"])
    kept: list[dict] = []
    for d in detections:
        dup = False
        for k in kept:
            merge_r = min(d["r"], k["r"])
            if (d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 < merge_r ** 2:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept
