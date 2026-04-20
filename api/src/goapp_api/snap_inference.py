"""Run the synth-trained snap-to-grid regressor on a tight-ish board crop."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .snap_classifier import IMG_SIZE

log = logging.getLogger(__name__)
from .paths import SNAP_CLASSIFIER_PATH as MODEL_PATH  # noqa: E402


class SnapModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise SnapModelNotLoaded(f"model file not found: {MODEL_PATH}")
    import torch

    from .snap_classifier import SnapToGrid
    log.info("loading snap classifier from %s", MODEL_PATH)
    device = _pick_device()
    model = SnapToGrid()
    state = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, device


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_snap(crop_bgr: np.ndarray) -> dict[str, float]:
    """Return {pitch_x, pitch_y, origin_x, origin_y} as fractions of the
    crop's width/height. At inference, multiply pitch_x by crop_width to
    get pitch in pixels in the original (un-resized) crop."""
    import torch

    model, device = _load_model()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = (torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        out = model(tensor)[0].cpu().numpy()
    return {
        "pitch_x": float(out[0]),
        "pitch_y": float(out[1]),
        "origin_x": float(out[2]),
        "origin_y": float(out[3]),
    }
