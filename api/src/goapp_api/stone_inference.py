"""Stone detection via the trained YOLOv8 detector.

Two classes: 0 = B (black), 1 = W (white). Each prediction is a bbox
around a stone; we take the bbox center as the stone position.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

log = logging.getLogger(__name__)

from .paths import STONE_DETECTOR_PATH as MODEL_PATH  # noqa: E402

PEAK_THRESH = 0.3  # kept as the `peak_thresh` kwarg name for API compat
TRAIN_IMG_SIZE = 640  # training imgsz; used as default for larger crops


class StoneModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise StoneModelNotLoaded(f"model file not found: {MODEL_PATH}")
    from ultralytics import YOLO
    log.info("loading stone YOLO from %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    return model


def detect_stones_cnn(
    crop_bgr: np.ndarray,
    peak_thresh: float = PEAK_THRESH,
) -> list[dict]:
    """Run YOLO on a board crop; return detected stone centers.

    Each entry: {"x", "y", "r", "color", "conf"} in the crop's pixel
    coordinate space. `peak_thresh` is used as the YOLO confidence
    threshold (kept as kwarg name for API compatibility).
    """
    model = _load_model()
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    # The model was trained on per-board crops at imgsz=640. Small crops
    # (e.g. cho-chikun 336x136) have stones at ~9px radius which is below
    # the training distribution — upscaling to 640 brings them into range.
    # Very large crops (>640) are downscaled to 640 as usual.
    imgsz = TRAIN_IMG_SIZE

    results = model.predict(
        crop_bgr,
        imgsz=imgsz,
        conf=float(peak_thresh),
        iou=0.5,
        augment=True,  # test-time aug: multi-scale + flip, merged via NMS
        verbose=False,
    )
    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    # xyxy in original-image pixels; cls in {0: B, 1: W}; conf in [0, 1]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()

    # Post-classify color from the actual pixel values at each detection
    # center. Pixel darkness is unambiguous even when YOLO's class head
    # gets confused on lower-contrast scans.
    gray_img = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )

    detections: list[dict] = []
    for (x0, y0, x1, y1), c, p in zip(xyxy, cls, conf):
        cx = float((x0 + x1) / 2.0)
        cy = float((y0 + y1) / 2.0)
        r = float(max(x1 - x0, y1 - y0) / 2.0)
        color = "B" if int(c) == 0 else "W"
        # Sample the center 1/3 of the bbox — avoids grid lines at stone
        # edges and captures the stone's actual fill color.
        inner = max(1, int(r * 0.33))
        ix0 = max(0, int(cx - inner))
        ix1 = min(gray_img.shape[1], int(cx + inner) + 1)
        iy0 = max(0, int(cy - inner))
        iy1 = min(gray_img.shape[0], int(cy + inner) + 1)
        if ix1 > ix0 and iy1 > iy0:
            mean_gray = float(gray_img[iy0:iy1, ix0:ix1].mean())
            if mean_gray < 100:
                color = "B"
            elif mean_gray > 180:
                color = "W"
        detections.append({
            "x": cx,
            "y": cy,
            "r": r,
            "color": color,
            "conf": float(p),
        })

    # Deduplicate: TTA can leave duplicates across scales. True duplicates
    # sit within ~0.2·pitch of each other; adjacent-cell stones are a
    # full pitch apart. A threshold of half the smaller radius (since
    # r ≈ 0.4·pitch, this is ~0.2·pitch) keeps adjacent stones separate
    # while collapsing overlapping detections of the same stone.
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
