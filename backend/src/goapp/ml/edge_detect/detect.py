"""Classical edge detection on board crops.

Decides which sides of a bbox crop are real board boundaries (the outer
frame of the 19x19) vs. interior grid rows/columns near the crop edge.

Method per side:
  1. Adaptive-threshold the crop so even faint interior grid lines survive
     (Otsu or a fixed threshold throws them out in scans).
  2. Take a narrow strip along that side. Project onto a 1D density
     profile (dark-pixel fraction per row of the strip).
  3. Find the first two contiguous peaks: peak[0] = outermost line
     (frame candidate), peak[1] = first interior line (baseline).
  4. Score = how frame-like the outer peak is *relative* to the inner
     baseline, combining:
       - thickness ratio (hm2 frames are drawn thicker than grid lines)
       - darkness delta (cho-chikun frames are drawn darker than grid lines)
     max() of the two — either signal is enough.
"""

from __future__ import annotations

import cv2
import numpy as np


# Crops smaller than this in either dimension can't carry a meaningful
# strip + peak signal, so we skip them.
MIN_CROP_SIZE = 30

# Adaptive-threshold params — the values that pick up faint scanned grid
# lines without melting frames into the background. Same family of values
# is used in pitch/measure.py; kept duplicated since the two modules score
# different things and we don't want a coupling that pretends they don't.
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8

# Strip width (px) along each candidate side. The strip needs to be tall
# enough to clear ~2 grid pitches at scan resolution. We take the larger
# of a fixed floor and 1/3 of the side length.
STRIP_BAND_MIN = 100

# Peak threshold inside the strip: peaks must reach at least PEAK_FRAC of
# the strip's max density, with a floor that prevents low-contrast strips
# (mostly empty) from generating false peaks. Tuned across hm2 (faint
# scans, max ~0.5) and cho-chikun (high-contrast, stone-stacking noise);
# see the body of _score_side for the failure modes.
PEAK_THRESH_FRAC = 0.5
PEAK_THRESH_FLOOR = 0.25

# Darkness-delta normalization: a 30-gray gap between outer and inner
# peak medians is treated as full-confidence "frame is darker".
DARKNESS_DELTA_NORM = 30.0


class EdgeModelNotLoaded(RuntimeError):
    """Retained for API compatibility. Classical detector never raises this."""


def model_available() -> bool:
    return True


def detect_edges(
    crop_bgr: np.ndarray, threshold: float = 0.5,
) -> dict[str, bool]:
    probs = detect_edges_with_probs(crop_bgr)
    return {k: v >= threshold for k, v in probs.items()}


def detect_edges_with_probs(crop_bgr: np.ndarray) -> dict[str, float]:
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )
    h, w = gray.shape
    if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
        return {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}

    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )

    return {
        "top":    _score_side(gray, bi, "top"),
        "bottom": _score_side(gray, bi, "bottom"),
        "left":   _score_side(gray, bi, "left"),
        "right":  _score_side(gray, bi, "right"),
    }


def _score_side(gray: np.ndarray, bi: np.ndarray, side: str) -> float:
    if side == "top":
        g, b = gray, bi
    elif side == "bottom":
        g, b = gray[::-1], bi[::-1]
    elif side == "left":
        g, b = gray.T, bi.T
    elif side == "right":
        g, b = gray.T[::-1], bi.T[::-1]
    else:
        return 0.0

    H, W = b.shape
    band = min(H, max(STRIP_BAND_MIN, H // 3))
    strip_b = b[:band]
    strip_g = g[:band]

    density = strip_b.mean(axis=1) / 255.0
    # Adaptive threshold: a fraction of the strip's max density, with a
    # floor. See PEAK_THRESH_FRAC / PEAK_THRESH_FLOOR for why these
    # specific values.
    thr = max(PEAK_THRESH_FLOOR, density.max() * PEAK_THRESH_FRAC)
    peaks = _find_runs(density, min_density=thr)
    if len(peaks) < 2:
        return 0.0

    outer_start, outer_end, _ = peaks[0]
    inner_start, inner_end, _ = peaks[1]

    outer_thick = outer_end - outer_start + 1
    inner_thick = inner_end - inner_start + 1

    # Median gray of the dark pixels in each peak's span.
    outer_gray = _median_peak_gray(strip_g, strip_b, outer_start, outer_end)
    inner_gray = _median_peak_gray(strip_g, strip_b, inner_start, inner_end)
    if outer_gray is None or inner_gray is None:
        return 0.0

    # Relative thickness: frames are thicker. 1.0 ratio → 0, 2.0 → 1.0.
    thick_score = max(0.0, min(1.0, (outer_thick / max(inner_thick, 1)) - 1.0))
    # Relative darkness: frames are darker (lower gray).
    dark_score = max(0.0, min(1.0, (inner_gray - outer_gray) / DARKNESS_DELTA_NORM))

    return max(thick_score, dark_score)


def _median_peak_gray(
    strip_g: np.ndarray, strip_b: np.ndarray, start: int, end: int,
) -> float | None:
    rows_g = strip_g[start:end + 1]
    rows_b = strip_b[start:end + 1]
    mask = rows_b > 0
    if not mask.any():
        return None
    return float(np.median(rows_g[mask]))


def _find_runs(
    profile: np.ndarray, min_density: float,
) -> list[tuple[int, int, float]]:
    runs: list[tuple[int, int, float]] = []
    n = len(profile)
    i = 0
    while i < n:
        if profile[i] >= min_density:
            s = i
            while i < n and profile[i] >= min_density:
                i += 1
            runs.append((s, i - 1, (s + i - 1) / 2.0))
        else:
            i += 1
    return runs
