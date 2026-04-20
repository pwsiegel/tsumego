"""Run the synth-trained edge classifier on a board crop."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .edge_classifier import EDGE_NAMES, IMG_SIZE

log = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "edge_classifier.pt"


class EdgeModelNotLoaded(RuntimeError):
    pass


def model_available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise EdgeModelNotLoaded(f"model file not found: {MODEL_PATH}")
    import torch

    from .edge_classifier import EdgeClassifier
    log.info("loading edge classifier from %s", MODEL_PATH)
    device = _pick_device()
    model = EdgeClassifier()
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


def detect_edges(
    crop_bgr: np.ndarray, threshold: float = 0.5,
) -> dict[str, bool]:
    """Return {"left","right","top","bottom": bool}: which crop edges are
    real board boundaries. All-False when the model is unsure about every
    edge."""
    import torch

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
        logits = model(tensor)[0].cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {name: bool(probs[i] >= threshold) for i, name in enumerate(EDGE_NAMES)}


def detect_edges_with_probs(crop_bgr: np.ndarray) -> dict[str, float]:
    import torch

    model, device = _load_model()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = (torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0].cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {name: float(probs[i]) for i, name in enumerate(EDGE_NAMES)}
