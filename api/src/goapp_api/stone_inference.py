"""Stone detection via the trained UNet heatmap regressor."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "stone_detector.pt"
IMG_SIZE = 512
PEAK_THRESH = 0.3
NMS_RADIUS_FRAC = 0.02  # relative to IMG_SIZE


class StoneModelNotLoaded(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise StoneModelNotLoaded(f"model file not found: {MODEL_PATH}")
    import torch
    from .train_stones import UNet

    log.info("loading stone UNet from %s", MODEL_PATH)
    device = _pick_device()
    model = UNet()
    state = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, device


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_available() -> bool:
    return MODEL_PATH.exists()


def detect_stones_cnn(
    crop_bgr: np.ndarray,
    peak_thresh: float = PEAK_THRESH,
) -> list[dict]:
    """Run the UNet on a board crop; return detected stone centers.

    Each entry: {"x", "y", "r", "color", "conf"} in original crop pixel space.
    """
    import torch

    model, device = _load_model()
    orig_h, orig_w = crop_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = (
        torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    ).to(device)

    with torch.no_grad():
        heat = model(tensor)[0].cpu().numpy()  # (2, H, W)

    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE
    nms_r = max(3, int(IMG_SIZE * NMS_RADIUS_FRAC))

    detections: list[dict] = []
    for ch, color in enumerate(("B", "W")):
        peaks = _extract_peaks(heat[ch], peak_thresh, nms_r)
        for px, py, conf in peaks:
            detections.append({
                "x": float(px * sx),
                "y": float(py * sy),
                "r": 0.0,  # filled in below, once we know the stone pitch
                "color": color,
                "conf": float(conf),
            })
    display_r = _estimate_display_radius(
        detections,
        fallback=min(orig_h, orig_w) * 0.025,
        image_max=min(orig_h, orig_w),
    )
    for d in detections:
        d["r"] = display_r
    # Same physical stone can spike both channels — most often when a white
    # number/mark is drawn inside a black stone (or vice versa), since the
    # inverted-color glyph genuinely looks like a small version of the
    # opposite color. Merge aggressively across channels and keep the
    # higher-confidence detection. The merge distance is floored at a
    # fraction of the crop size so it can't collapse when nearest-neighbor
    # distances themselves are tiny (which happens exactly when duplicates
    # dominate).
    detections.sort(key=lambda s: -s["conf"])
    deduped: list[dict] = []
    merge_dist = max(
        display_r * 1.6,
        min(orig_h, orig_w) * 0.035,
        nms_r * max(sx, sy),
    )
    for d in detections:
        if any(
            (d["x"] - m["x"]) ** 2 + (d["y"] - m["y"]) ** 2 < merge_dist ** 2
            for m in deduped
        ):
            continue
        deduped.append(d)
    return deduped


def _extract_peaks(
    heatmap: np.ndarray, thresh: float, nms_radius: int,
) -> list[tuple[float, float, float]]:
    """Non-max-suppressed peaks above `thresh`. Returns (x, y, conf) list."""
    ksize = nms_radius * 2 + 1
    dilated = cv2.dilate(heatmap, np.ones((ksize, ksize), np.uint8))
    is_peak = (heatmap == dilated) & (heatmap >= thresh)
    ys, xs = np.where(is_peak)
    out = [(float(x), float(y), float(heatmap[y, x])) for x, y in zip(xs, ys)]
    out.sort(key=lambda p: -p[2])
    return out


def _estimate_display_radius(
    detections: list[dict], fallback: float, image_max: float | None = None,
) -> float:
    """Guess a reasonable drawing radius from the typical stone-to-nearest-
    neighbor distance (≈ one grid pitch; stones are ~45% of a pitch).
    Clamped from above so sparse boards don't produce absurdly large circles:
    a stone can't be bigger than ~10% of the shorter crop dimension on a
    19x19 board."""
    if len(detections) < 2:
        return max(fallback, 6.0)
    xs = np.array([d["x"] for d in detections])
    ys = np.array([d["y"] for d in detections])
    nearest_dists: list[float] = []
    for i in range(len(detections)):
        dx = xs - xs[i]
        dy = ys - ys[i]
        d2 = dx * dx + dy * dy
        d2[i] = np.inf
        nearest_dists.append(float(np.sqrt(d2.min())))
    typical = float(np.median(nearest_dists))
    r = typical * 0.4
    if image_max is not None:
        r = min(r, image_max * 0.1)
    return max(6.0, r)
