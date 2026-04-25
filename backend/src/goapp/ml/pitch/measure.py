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
    if h < 40 or w < 40:
        return empty

    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25, C=8,
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
    band = min(H, max(150, H // 3))
    strip = s[:band]

    density = strip.mean(axis=1) / 255.0
    thr = max(0.25, density.max() * 0.5)
    peaks = _find_runs(density, min_density=thr)
    if len(peaks) < 2:
        return None, None, None
    _, _, c0 = peaks[0]
    _, _, c1 = peaks[1]
    pitch = c1 - c0
    if pitch < 10:
        return None, None, None

    # Walk the periodic grid from the frame across the FULL axis. For
    # each predicted line position c0 + k*pitch, check whether there's
    # a density peak within ±pitch/3 of that position. Individual grid
    # lines can dip below threshold (stones sitting on them, local
    # noise), so allow up to 2 consecutive misses before concluding
    # we've run off the board.
    full_density = s.mean(axis=1) / 255.0
    full_thr = max(0.2, full_density.max() * 0.35)
    last_line = c0
    misses = 0
    for k in range(1, 19):
        predicted = c0 + k * pitch
        if predicted >= H:
            break
        lo = max(0, int(predicted - pitch / 3))
        hi = min(H, int(predicted + pitch / 3) + 1)
        if full_density[lo:hi].max() >= full_thr:
            last_line = predicted
            misses = 0
        else:
            misses += 1
            if misses >= 2:
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
