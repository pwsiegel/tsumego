"""Shared YOLOv8 inference via onnxruntime.

Replaces the ultralytics runtime dep in the serving image. Pre/post
processing matches ultralytics' default detect pipeline: letterbox to
imgsz x imgsz, BGR→RGB→[0,1] CHW, then classic xywh-center decode +
NMS on the (1, 4+C, N) output.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    cls: int
    conf: float


def _letterbox(
    img_bgr: np.ndarray, size: int, pad_value: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """Resize + pad to (size, size), preserving aspect.

    Returns (canvas, scale, pad_x, pad_y) where scale is the resize factor
    applied to the original image and (pad_x, pad_y) is the top-left of
    the resized image inside the canvas.
    """
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _preprocess(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    canvas, scale, pad_x, pad_y = _letterbox(img_bgr, size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return chw[None, ...], scale, pad_x, pad_y


@lru_cache(maxsize=4)
def _session(onnx_path: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def predict(
    onnx_path: Path,
    img_bgr: np.ndarray,
    *,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int = 640,
) -> list[Detection]:
    """Run YOLOv8 detection; return boxes in original-image coords."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return []
    sess = _session(str(onnx_path))
    inp, scale, pad_x, pad_y = _preprocess(img_bgr, imgsz)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: inp})[0]  # (1, 4+C, N)

    pred = out[0].T  # (N, 4+C)
    if pred.shape[0] == 0:
        return []
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    keep = confs >= conf_thresh
    if not keep.any():
        return []
    boxes_xywh = boxes_xywh[keep]
    cls_ids = cls_ids[keep]
    confs = confs[keep]

    # cv2.dnn.NMSBoxes wants top-left xywh.
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0
    nms_boxes = np.stack([x0, y0, bw, bh], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), conf_thresh, iou_thresh)
    if len(idxs) == 0:
        return []
    idxs = np.asarray(idxs).reshape(-1)

    out_dets: list[Detection] = []
    for i in idxs:
        bx0 = (x0[i] - pad_x) / scale
        by0 = (y0[i] - pad_y) / scale
        bx1 = bx0 + bw[i] / scale
        by1 = by0 + bh[i] / scale
        bx0 = float(max(0.0, min(w, bx0)))
        by0 = float(max(0.0, min(h, by0)))
        bx1 = float(max(0.0, min(w, bx1)))
        by1 = float(max(0.0, min(h, by1)))
        out_dets.append(Detection(
            x0=bx0, y0=by0, x1=bx1, y1=by1,
            cls=int(cls_ids[i]),
            conf=float(confs[i]),
        ))
    return out_dets


def model_available(onnx_path: Path) -> bool:
    return onnx_path.exists()
