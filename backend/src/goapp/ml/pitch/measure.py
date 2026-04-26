"""Grid-pitch measurement directly from the crop image.

Measures the cell size (pitch) by looking at the image and finding the
distance between the frame line and the first interior grid line parallel
to it — the same thing a human does when reading a Go diagram.

This sidesteps the "approximate GCD of pairwise stone distances" problem
the stone-based estimator was trying to solve. Pitch is a local geometric
property of the printed grid; we measure it where we have the cleanest
signal (near the frame, where curvature is minimal on scans).

Method:
  1. Adaptive-threshold the crop. Robust to paper brightness variation
     and picks up even the faint interior grid lines that a fixed
     threshold would miss.
  2. For each side whose frame was confirmed by the edge detector,
     rotate so that side is on top, extract a narrow strip, and compute
     the per-row dark-pixel density (1D profile).
  3. First contiguous peak = frame. Second peak = first interior grid
     line. Pitch = distance between their centers.
  4. Take the median across available sides.
"""

from __future__ import annotations

import cv2
import numpy as np


# Crops smaller than this in either dimension can't carry a meaningful
# strip + peak signal.
MIN_CROP_SIZE = 40

# Adaptive-threshold params; same family as edge_detect/detect.py. Kept
# duplicated so the two modules can drift independently if their needs
# diverge.
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8

# Per-side strip width for finding the first two grid peaks. Wider than
# edge_detect's strip because we need to clear at least 2 grid pitches.
STRIP_BAND_MIN = 150

# Peak threshold inside the per-side strip (frame + first interior line).
STRIP_PEAK_FRAC = 0.5
STRIP_PEAK_FLOOR = 0.25

# Smaller pitches than this aren't physical (5px boards don't exist) and
# usually mean we picked up noise as the second peak.
MIN_PITCH_PIXELS = 10

# Threshold for the *full-axis* density profile we use to walk the
# periodic grid inward from the frame. Looser than the per-side strip's
# threshold because individual interior grid lines can be partially
# occluded by stones; we want to catch them anyway.
WALK_PEAK_FRAC = 0.35
WALK_PEAK_FLOOR = 0.2

# How far each predicted grid-line position can drift before we declare a
# miss, expressed as a fraction of pitch. ±pitch/3 tolerates a fair amount
# of perspective without overlapping the next predicted line.
WALK_TOLERANCE_FRAC = 1 / 3

# Number of consecutive missed grid predictions before we conclude we've
# walked off the board. Allows the occasional stone-occluded line.
WALK_MAX_MISSES = 2


def measure_grid(
    crop_bgr: np.ndarray, edges: dict[str, bool],
) -> dict:
    """Measure grid pitch + frame position + far-side grid-line extent
    per confirmed edge.

    Returns:
        {
            "pitch": float | None,            # median of per-side pitches
            "top":    frame_y_pixel | None,
            "bottom": frame_y_pixel | None,
            "left":   frame_x_pixel | None,
            "right":  frame_x_pixel | None,
            "top_pitch" … "right_pitch": per-side pitches | None,
            "top_last":    last visible grid-line y (scanning down from top) | None,
            "bottom_last": last visible grid-line y (scanning up from bottom) | None,
            "left_last":   last visible grid-line x (scanning right from left) | None,
            "right_last":  last visible grid-line x (scanning left from right) | None,
        }
    """
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )
    h, w = gray.shape
    empty = {
        "pitch": None,
        "top": None, "bottom": None, "left": None, "right": None,
        "top_pitch": None, "bottom_pitch": None,
        "left_pitch": None, "right_pitch": None,
        "top_last": None, "bottom_last": None,
        "left_last": None, "right_last": None,
    }
    if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
        return empty

    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )

    result = dict(empty)
    pitches: list[float] = []
    for side in ("top", "bottom", "left", "right"):
        if not edges.get(side, False):
            continue
        pitch, frame_local, last_local = _measure_side(bi, side)
        if pitch is None:
            continue
        # De-rotate back into original crop coords. frame_local and
        # last_local are measured from the rotated top → positive values.
        if side == "top":
            result["top"] = frame_local
            result["top_last"] = last_local
        elif side == "bottom":
            result["bottom"] = (h - 1) - frame_local
            result["bottom_last"] = (h - 1) - last_local
        elif side == "left":
            result["left"] = frame_local
            result["left_last"] = last_local
        elif side == "right":
            result["right"] = (w - 1) - frame_local
            result["right_last"] = (w - 1) - last_local
        result[f"{side}_pitch"] = pitch
        pitches.append(pitch)

    result["pitch"] = float(np.median(pitches)) if pitches else None
    return result


def _measure_side(
    bi: np.ndarray, side: str,
) -> tuple[float | None, float | None, float | None]:
    """Return (pitch, frame_position, last_line_position). Frame is the
    outermost peak; last_line is the farthest peak along the periodic
    grid sequence found by walking inward from the frame."""
    if side == "top":
        s = bi
    elif side == "bottom":
        s = bi[::-1]
    elif side == "left":
        s = bi.T
    elif side == "right":
        s = bi.T[::-1]
    else:
        return None, None, None

    H, W = s.shape
    band = min(H, max(STRIP_BAND_MIN, H // 3))
    strip = s[:band]

    density = strip.mean(axis=1) / 255.0
    thr = max(STRIP_PEAK_FLOOR, density.max() * STRIP_PEAK_FRAC)
    peaks = _find_runs(density, min_density=thr)
    if len(peaks) < 2:
        return None, None, None
    _, _, c0 = peaks[0]
    _, _, c1 = peaks[1]
    pitch = c1 - c0
    if pitch < MIN_PITCH_PIXELS:
        return None, None, None

    # Walk the periodic grid from the frame across the FULL axis. For
    # each predicted line position c0 + k*pitch, check whether there's
    # a density peak within ±WALK_TOLERANCE_FRAC*pitch of it. Individual
    # grid lines can dip below threshold (stones sitting on them, local
    # noise), so allow up to WALK_MAX_MISSES consecutive misses before
    # concluding we've run off the board.
    full_density = s.mean(axis=1) / 255.0
    full_thr = max(WALK_PEAK_FLOOR, full_density.max() * WALK_PEAK_FRAC)
    tol = pitch * WALK_TOLERANCE_FRAC
    last_line = c0
    misses = 0
    for k in range(1, 19):
        predicted = c0 + k * pitch
        if predicted >= H:
            break
        lo = max(0, int(predicted - tol))
        hi = min(H, int(predicted + tol) + 1)
        if full_density[lo:hi].max() >= full_thr:
            last_line = predicted
            misses = 0
        else:
            misses += 1
            if misses >= WALK_MAX_MISSES:
                break

    return pitch, c0, last_line


def _find_runs(
    profile: np.ndarray, min_density: float,
) -> list[tuple[int, int, float]]:
    """Return [(start, end, center), ...] for contiguous runs of
    profile >= min_density."""
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
