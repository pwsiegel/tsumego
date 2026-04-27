"""Fit a 2D lattice to detected intersection + stone centers.

Each detection should sit on the visible portion of a 19×19 grid. The
fitter finds (pitch_x, pitch_y, origin_x, origin_y) such that every point
projects to integer (col, row), then determines which sides of the crop
hit the actual board edge (vs. a window cut-off).

Algorithm per axis:
  1. Pairwise differences of all coords → 1D histogram → local maxima
     are candidate pitches.
  2. For each candidate, grid-search the origin offset to find the inlier
     count (points within SNAP_TOL_FRAC·pitch of an integer lattice line).
  3. Keep the (pitch, origin) with the most inliers.
  4. Refine pitch + origin via least-squares regression on inliers. This
     also rejects the candidate "2·pitch" since its inlier count is
     halved.
  5. Normalize origin so col_local=0 corresponds to the smallest occupied
     column.

A side is the real board edge iff the outermost lattice line is within
EDGE_THRESH_FRAC·pitch of the crop boundary — no room for another
intersection past it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PITCH_MIN = 8.0
PITCH_MAX = 100.0
SNAP_TOL_FRAC = 0.25
EDGE_THRESH_FRAC = 0.6
MIN_POINTS = 8


@dataclass(frozen=True)
class Lattice:
    pitch_x: float
    pitch_y: float
    origin_x: float          # page-pixel x of (col_local=0)
    origin_y: float          # page-pixel y of (row_local=0)
    n_cols: int              # max col_local + 1 (visible cols)
    n_rows: int              # max row_local + 1 (visible rows)
    edges: dict[str, bool]   # {"left","right","top","bottom"}


def _candidate_pitches(coords: np.ndarray) -> list[float]:
    """1D histogram of pairwise differences; return all local maxima as
    candidate pitches, ordered by histogram peak height (tallest first)."""
    n = len(coords)
    diffs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(float(coords[j] - coords[i]))
            if PITCH_MIN <= d <= PITCH_MAX:
                diffs.append(d)
    if len(diffs) < 4:
        return []
    arr = np.asarray(diffs)
    bins = np.arange(PITCH_MIN, PITCH_MAX + 1.0, 1.0)
    counts, _ = np.histogram(arr, bins=bins)
    kernel = np.array([1, 2, 4, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((float(smoothed[i]), float(bins[i]) + 0.5))
    peaks.sort(reverse=True)
    return [p for _, p in peaks]


def _score_pitch(coords: np.ndarray, pitch: float) -> tuple[int, float]:
    """For a candidate pitch, find the origin with the most inliers."""
    best_count = 0
    best_origin = float(coords.min())
    n_offsets = max(20, int(round(pitch)))
    for offset_frac in np.linspace(0, 1, n_offsets, endpoint=False):
        origin = float(coords.min()) + offset_frac * pitch
        residuals = (coords - origin) % pitch
        residuals = np.minimum(residuals, pitch - residuals)
        count = int((residuals < SNAP_TOL_FRAC * pitch).sum())
        if count > best_count:
            best_count = count
            best_origin = origin
    return best_count, best_origin


def _refine(coords: np.ndarray, pitch: float, origin: float) -> tuple[float, float]:
    """Least-squares refine pitch + origin using the inlier subset."""
    cols = np.round((coords - origin) / pitch).astype(int)
    residuals = np.abs(coords - (origin + cols * pitch))
    inlier_mask = residuals < SNAP_TOL_FRAC * pitch
    if inlier_mask.sum() < 4:
        return pitch, origin
    x_in = coords[inlier_mask]
    c_in = cols[inlier_mask].astype(float)
    A = np.vstack([c_in, np.ones(len(c_in))]).T
    p_ref, o_ref = np.linalg.lstsq(A, x_in, rcond=None)[0]
    if PITCH_MIN <= p_ref <= PITCH_MAX:
        return float(p_ref), float(o_ref)
    return pitch, origin


def _estimate_axis(coords: np.ndarray) -> tuple[float, float] | None:
    """Estimate (pitch, origin) for one axis. Origin is the page-pixel
    coordinate of col_local=0 (smallest occupied integer column)."""
    if len(coords) < 4:
        return None
    candidates = _candidate_pitches(coords)
    if not candidates:
        return None
    best_count = 0
    best_pitch = None
    best_origin = None
    for p in candidates:
        count, origin = _score_pitch(coords, p)
        if count > best_count:
            best_count = count
            best_pitch = p
            best_origin = origin
    if best_pitch is None or best_count < 4:
        return None
    pitch, origin = _refine(coords, best_pitch, best_origin)
    cols = np.round((coords - origin) / pitch).astype(int)
    col_min = int(cols.min())
    origin = origin + col_min * pitch
    return pitch, float(origin)


def fit_lattice(
    points: list[tuple[float, float]],
    crop_w: int,
    crop_h: int,
) -> Lattice | None:
    """Fit a 2D lattice to detected (intersection or stone) centers.
    Returns None if there aren't enough points or the fit fails."""
    if len(points) < MIN_POINTS:
        return None
    pts = np.asarray(points, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]

    fx = _estimate_axis(xs)
    fy = _estimate_axis(ys)
    if fx is None or fy is None:
        return None
    pitch_x, origin_x = fx
    pitch_y, origin_y = fy

    cols = np.round((xs - origin_x) / pitch_x).astype(int)
    rows = np.round((ys - origin_y) / pitch_y).astype(int)
    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1

    left_x = origin_x
    right_x = origin_x + (n_cols - 1) * pitch_x
    top_y = origin_y
    bottom_y = origin_y + (n_rows - 1) * pitch_y

    edges = {
        "left": left_x < EDGE_THRESH_FRAC * pitch_x,
        "right": (crop_w - right_x) < EDGE_THRESH_FRAC * pitch_x,
        "top": top_y < EDGE_THRESH_FRAC * pitch_y,
        "bottom": (crop_h - bottom_y) < EDGE_THRESH_FRAC * pitch_y,
    }

    return Lattice(
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        n_cols=n_cols,
        n_rows=n_rows,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Quick self-test: synthetic lattice with noise + outliers.
# Run: uv --directory backend run python -m goapp.ml.intersection_detect.lattice
# ---------------------------------------------------------------------------

def _selftest() -> None:
    rng = np.random.default_rng(0)
    pitch_true = 23.7
    origin_x_true = 17.2
    origin_y_true = 14.5
    pts: list[tuple[float, float]] = []
    for c in range(12):
        for r in range(9):
            x = origin_x_true + c * pitch_true + rng.normal(0, 0.7)
            y = origin_y_true + r * pitch_true + rng.normal(0, 0.7)
            pts.append((x, y))
    # Sprinkle outliers
    for _ in range(8):
        pts.append((rng.uniform(0, 300), rng.uniform(0, 300)))
    crop_w = int(origin_x_true + 11 * pitch_true + 5)  # tight on right edge
    crop_h = int(origin_y_true + 8 * pitch_true + 30)  # cut off on bottom
    lat = fit_lattice(pts, crop_w, crop_h)
    print(f"true:  pitch={pitch_true}  origin=({origin_x_true:.2f},{origin_y_true:.2f})")
    print(f"fit :  {lat}")


if __name__ == "__main__":
    _selftest()
