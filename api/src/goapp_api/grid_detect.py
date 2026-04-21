"""Classical grid detector for a tight board crop.

Two-stage approach:
  1. Morphological line extraction — opening the binarized image with a
     long-axis kernel keeps only sustained horizontal/vertical lines.
     Stones, text, and other clutter don't survive this filter, so the
     projection signal is almost purely grid.
  2. Periodicity-aware fit — search (pitch, origin) pairs to maximize
     the total line-mask signal at evenly-spaced positions. Robust to
     missing lines (stones covering part of a line only dent the signal
     at that row; the rest of the comb is intact).

No training. No gradients. Deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BOARD_SIZE = 19


@dataclass(frozen=True)
class GridDetectResult:
    pitch_x_px: float | None
    pitch_y_px: float | None
    origin_x_px: float | None
    origin_y_px: float | None
    vert_xs: list[float]
    horz_ys: list[float]


def detect_grid(crop_bgr: np.ndarray) -> GridDetectResult:
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    col_signal, row_signal = _line_projections(gray, w, h)

    # Search range: floor at w/22 (allow a ~22-interval full-board crop);
    # ceiling at w/3.5 (allow top-/bottom-strip crops with as few as 3-4
    # visible rows, where pitch is a big fraction of the short dimension).
    min_px = max(8.0, w / 22.0); max_px = w / 3.5
    min_py = max(8.0, h / 22.0); max_py = h / 3.5

    pitch_x, origin_x, score_x = _fit_comb(col_signal, min_px, max_px)
    pitch_y, origin_y, score_y = _fit_comb(row_signal, min_py, max_py)

    # Squareness sanity check. Printed Go grids are square: if the two
    # axes disagree by more than 15%, trust whichever had the stronger
    # fit signal and use its pitch for the other axis (refit only the
    # origin on the disagreeing axis).
    if (pitch_x is not None and pitch_y is not None
            and max(pitch_x, pitch_y) / min(pitch_x, pitch_y) > 1.15):
        if score_x >= score_y:
            pitch_y, origin_y, _ = _fit_origin(row_signal, pitch_x)
        else:
            pitch_x, origin_x, _ = _fit_origin(col_signal, pitch_y)

    # Enumerate comb positions, then drop any whose local signal is weak
    # — those are positions where the fit extrapolated past the actual
    # grid into empty paper or caption text.
    vert_xs = _active_comb_positions(col_signal, origin_x, pitch_x, w)
    horz_ys = _active_comb_positions(row_signal, origin_y, pitch_y, h)

    return GridDetectResult(
        pitch_x_px=pitch_x, pitch_y_px=pitch_y,
        origin_x_px=origin_x, origin_y_px=origin_y,
        vert_xs=vert_xs, horz_ys=horz_ys,
    )


def _line_projections(gray: np.ndarray, w: int, h: int):
    """Produce 1D signals whose peaks are at grid-line positions.

    Uses morphological opening on the adaptive-thresholded image with a
    horizontal/vertical kernel. This keeps only features that are
    sustained along that axis — i.e., grid lines — and rejects stones,
    text, labels, and other clutter.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )
    hk = max(10, w // 30)
    vk = max(10, h // 30)
    h_mask = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)),
    )
    v_mask = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)),
    )
    h_mask = h_mask.astype(np.float32) / 255.0
    v_mask = v_mask.astype(np.float32) / 255.0
    col_signal = v_mask.sum(axis=0) / max(1, h)
    row_signal = h_mask.sum(axis=1) / max(1, w)
    return col_signal, row_signal


def _prepare_signal(signal: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (signal, baseline, hit_threshold).

    The threshold sits 25% of the way from baseline toward the max. This
    is robust across both sparse and dense peak distributions: even if the
    max is a thick border at 3× interior strength, 0.25 × (max − baseline)
    is still well below interior peak height, so interior lines still
    register as hits."""
    if signal.size == 0:
        return signal, 0.0, 0.0
    baseline = float(signal.mean())
    peak_est = float(signal.max())
    thresh = baseline + 0.25 * (peak_est - baseline)
    return signal, baseline, thresh


def _comb_score(signal: np.ndarray, baseline: float, thresh: float,
                pitch: float, origin: float, tolerance: int = 3) -> float:
    """Score = hits − 0.5·misses, where a position is a "hit" if any value
    within ±tolerance px exceeds `thresh`. Miss penalty punishes subharmonics
    (half-pitch combs hit every peak but also have N/2 in-between misses);
    hit count prefers the complete fit over super-sparse combs that only
    hit a couple of strong peaks."""
    W = len(signal)
    if thresh <= baseline or W < 10:
        return -np.inf
    idx = np.arange(origin, W, pitch).astype(int)
    idx = idx[idx < W]
    if len(idx) < 5:
        return -np.inf
    hits = 0
    for i in idx:
        lo = max(0, i - tolerance)
        hi = min(W, i + tolerance + 1)
        if float(signal[lo:hi].max()) > thresh:
            hits += 1
    misses = len(idx) - hits
    return float(hits - 0.5 * misses)


def _fit_comb(signal: np.ndarray, min_pitch: float, max_pitch: float):
    """Search (pitch, origin) to maximize grid-line coverage."""
    W = len(signal)
    if W < 10 or max_pitch <= min_pitch:
        return None, None, -np.inf
    signal, baseline, thresh = _prepare_signal(signal)

    best_score = -np.inf
    best: tuple[float, float] | None = None

    for pitch in np.arange(min_pitch, max_pitch + 0.01, 1.0):
        step = max(1, int(pitch / 40))
        for origin in range(0, int(np.ceil(pitch)), step):
            score = _comb_score(signal, baseline, thresh, pitch, origin)
            if score > best_score:
                best_score = score
                best = (float(pitch), float(origin))

    if best is None:
        return None, None, -np.inf

    p0, o0 = best
    pitches = np.arange(max(min_pitch, p0 - 1.5), min(max_pitch, p0 + 1.5) + 0.01, 0.1)
    origins = np.arange(max(0.0, o0 - 1.0), min(p0, o0 + 1.0) + 0.001, 0.1)
    for pitch in pitches:
        for origin in origins:
            score = _comb_score(signal, baseline, thresh, pitch, origin)
            if score > best_score:
                best_score = score
                best = (float(pitch), float(origin))

    return best[0], best[1], best_score


def _fit_origin(signal: np.ndarray, pitch: float):
    """Fit only the origin for a given pitch."""
    signal, baseline, thresh = _prepare_signal(signal)
    best_score = -np.inf
    best_origin = 0.0
    for origin in np.arange(0.0, pitch, 0.1):
        score = _comb_score(signal, baseline, thresh, pitch, origin)
        if score > best_score:
            best_score = score
            best_origin = float(origin)
    return float(pitch), best_origin, best_score


def _enumerate_lines(origin, pitch, dim):
    if origin is None or pitch is None:
        return []
    xs: list[float] = []
    k = 0
    while origin + k * pitch <= dim + 0.5:
        xs.append(float(origin + k * pitch))
        k += 1
    return xs


def _active_comb_positions(
    signal: np.ndarray,
    origin: float | None,
    pitch: float | None,
    dim: int,
    search_radius: int = 3,
) -> list[float]:
    """Enumerate comb positions then keep only those with a real line peak
    nearby. Threshold is derived via top-K peak estimation so it doesn't
    collapse to noise on sparse-peak signals."""
    if origin is None or pitch is None or signal.size == 0:
        return []
    _, baseline, thresh = _prepare_signal(signal)
    if thresh <= baseline:
        return []
    kept: list[float] = []
    k = 0
    while origin + k * pitch <= dim + 0.5:
        pos = origin + k * pitch
        idx = int(round(pos))
        k += 1
        lo = max(0, idx - search_radius)
        hi = min(len(signal), idx + search_radius + 1)
        if hi <= lo:
            continue
        if float(signal[lo:hi].max()) >= thresh:
            kept.append(float(pos))
    return kept
