"""Run the synth-trained grid classifier on a tight board crop."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .grid_classifier import BOARD_SIZE, IMG_SIZE

log = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "grid_classifier.pt"


class GridModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise GridModelNotLoaded(f"model file not found: {MODEL_PATH}")
    import torch

    from .grid_classifier import GridClassifier
    log.info("loading grid classifier from %s", MODEL_PATH)
    device = _pick_device()
    model = GridClassifier()
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


def predict_grid(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a tight board crop (pixels at outer intersections), predict
    the 19x19 grid. Returns:
      - grid : (19, 19) int array of 0=empty, 1=B, 2=W
      - probs: (3, 19, 19) float array of softmax probs per class
    """
    import torch
    import torch.nn.functional as F

    model, device = _load_model()
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = (
        torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    ).to(device)
    with torch.no_grad():
        logits = model(tensor)  # (1, 3, 19, 19)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    grid = probs.argmax(axis=0).astype(np.int8)
    return grid, probs
