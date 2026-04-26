"""Inference helper for the grid-geometry detector.

Loads the trained model on first call (lazy) and exposes ``detect_grid``
returning a ``GridGeometry`` with pitch, origin, and edge booleans in
crop coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from .dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from .model import GridDetector

from ...paths import GRID_DETECTOR_PATH


# A side is classified as a real edge when the predicted grid bbox sits
# within this fraction of a pitch from the corresponding crop boundary.
EDGE_FRAC_THRESHOLD = 0.5


class GridModelNotLoaded(RuntimeError):
    pass


@dataclass(frozen=True)
class GridGeometry:
    grid_bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) in crop px
    pitch_x: float                                  # pixels
    pitch_y: float                                  # pixels
    edges: dict[str, bool]                          # left/right/top/bottom


def model_path() -> Path:
    return GRID_DETECTOR_PATH


def model_available() -> bool:
    return GRID_DETECTOR_PATH.exists()


@lru_cache(maxsize=1)
def _load_model() -> tuple[GridDetector, torch.device]:
    if not GRID_DETECTOR_PATH.exists():
        raise GridModelNotLoaded(
            f"grid detector weights not found at {GRID_DETECTOR_PATH}"
        )
    device = torch.device("cpu")
    model = GridDetector(pretrained=False)
    state = torch.load(GRID_DETECTOR_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device


def _preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(
        crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.unsqueeze(0)


def detect_grid(
    crop_bgr: np.ndarray,
    edge_frac_threshold: float = EDGE_FRAC_THRESHOLD,
) -> GridGeometry:
    h, w = crop_bgr.shape[:2]
    model, device = _load_model()
    x = _preprocess(crop_bgr).to(device)
    with torch.no_grad():
        out = model(x).squeeze(0).cpu().numpy()
    gx0, gy0, gx1, gy1, px_n, py_n = out.tolist()

    bx0 = float(gx0 * w)
    by0 = float(gy0 * h)
    bx1 = float(gx1 * w)
    by1 = float(gy1 * h)
    pitch_x = float(px_n * w)
    pitch_y = float(py_n * h)

    edges = {
        "left":   bx0 < edge_frac_threshold * pitch_x,
        "right":  (w - bx1) < edge_frac_threshold * pitch_x,
        "top":    by0 < edge_frac_threshold * pitch_y,
        "bottom": (h - by1) < edge_frac_threshold * pitch_y,
    }
    return GridGeometry(
        grid_bbox=(bx0, by0, bx1, by1),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        edges=edges,
    )
