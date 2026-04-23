"""Corner detection via the trained YOLOv8 detector.

Four classes: 0 = TL, 1 = TR, 2 = BL, 3 = BR. Each prediction is a bbox
around a board corner; we take the bbox center as the corner position.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

from .paths import CORNER_DETECTOR_PATH as MODEL_PATH  # noqa: E402

CONF_THRESH = 0.5
TRAIN_IMG_SIZE = 640

CLASS_NAMES = {0: "tl", 1: "tr", 2: "bl", 3: "br"}


class CornerModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise CornerModelNotLoaded(f"model file not found: {MODEL_PATH}")
    from ultralytics import YOLO
    log.info("loading corner YOLO from %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    return model


def detect_corners(
    crop_bgr: np.ndarray,
    conf_thresh: float = CONF_THRESH,
) -> list[dict]:
    """Run YOLO on a board crop; return detected corner positions.

    Each entry: {"corner": "tl"|"tr"|"bl"|"br", "x", "y", "conf"} in
    the crop's pixel coordinate space. At most one detection per corner
    class (highest confidence wins).
    """
    model = _load_model()
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    max_dim = max(orig_h, orig_w)
    if max_dim <= TRAIN_IMG_SIZE:
        imgsz = max(320, ((max_dim + 31) // 32) * 32)
    else:
        imgsz = TRAIN_IMG_SIZE

    results = model.predict(
        crop_bgr,
        imgsz=imgsz,
        conf=float(conf_thresh),
        iou=0.5,
        verbose=False,
    )
    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()

    # Keep only the highest-confidence detection per corner class.
    best: dict[str, dict] = {}
    for (x0, y0, x1, y1), c, p in zip(xyxy, cls, conf):
        name = CLASS_NAMES.get(int(c))
        if name is None:
            continue
        cx = float((x0 + x1) / 2.0)
        cy = float((y0 + y1) / 2.0)
        if name not in best or float(p) > best[name]["conf"]:
            best[name] = {"corner": name, "x": cx, "y": cy, "conf": float(p)}

    return list(best.values())
