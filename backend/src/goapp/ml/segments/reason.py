"""Fused lattice estimator from Hough segments + stone/intersection centers.

This is the only lattice fitter the pipeline uses. Both signal sources
project to 1D coordinates that should land on the same regular grid:

  * Segment endpoints — horizontal segments contribute their midpoint y,
    vertical segments contribute their midpoint x. Where grid ink is
    visible, every grid line produces one or more segments.
  * Stone and intersection centers — each detection at (x, y) contributes
    its x to the vertical-line pool and its y to the horizontal-line
    pool. These sit on lattice points by construction (a stone is placed
    on an intersection).

Combining them is robust because the failure modes are uncorrelated:
segments are weak where stones occlude grid lines (paint-out voids),
and stones are absent on empty areas of the board where segments are
strongest.

Algorithm per axis (1D):
  1. Build the position pool from both sources.
  2. Pairwise-difference histogram over the pool → smoothed local
     maxima are candidate pitches (ordered by peak height).
  3. For each candidate, grid-search origin to maximize the count of
     positions within SNAP_TOL_FRAC·pitch of an integer lattice line.
  4. Pick the (pitch, origin) with the most inliers.
  5. Least-squares refine pitch + origin on the inlier subset. This
     step is what rejects the spurious 2·pitch candidate: doubling the
     pitch halves the inlier count, so its score never wins.

Adapted from the original fit_lattice in intersection_detect/lattice.py,
which only consumed CNN points; this fuses Hough segments in too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Segment


# Orientation tolerance for accepting a segment as horizontal/vertical.
# 15° absorbs scan rotation and printed-grid skew without admitting
# stone-rim arcs.
ORIENT_TOLERANCE_DEG = 15.0

# Pitch search bounds, in absolute pixels. Inherited from the previous
# fit_lattice: 8 px floor (sub-board), 100 px ceiling (no real Go
# diagram in our PDFs uses tighter or looser pitch).
PITCH_MIN = 8.0
PITCH_MAX = 120.0

# Inlier definition: a position is "on the lattice" if it sits within
# this fraction of pitch from a predicted lattice line.
SNAP_TOL_FRAC = 0.25

# Histogram smoothing kernel for candidate-pitch detection.
_SMOOTH_KERNEL = np.array([1, 2, 4, 2, 1], dtype=float)
_SMOOTH_KERNEL /= _SMOOTH_KERNEL.sum()

# Minimum positions per axis to attempt a fit.
MIN_POSITIONS = 4

# Minimum inliers for the best (pitch, origin) candidate to be trusted.
MIN_INLIERS = 4

# Lower bound for pitch as a multiple of median stone radius. Stones
# physically can't overlap, so pitch >= 2 * stone_r is a hard floor;
# our printed boards run pitch ≈ 2.2-2.5 * r, so 1.7 leaves a small
# safety margin for stones shrunk by ink bleed or detection noise.
# Without this floor, the half-pitch ambiguity (every point on an
# N-pitch grid sits on N/2 too) lets the fitter pick a pitch half
# the true value when ties are noise-driven.
PITCH_FLOOR_RADIUS_MULT = 1.7


@dataclass(frozen=True)
class AxisLattice:
    pitch: float | None
    origin: float | None


@dataclass(frozen=True)
class FusedLattice:
    x: AxisLattice
    y: AxisLattice


def fit_lattice_fused(
    segments: list[Segment],
    stone_centers: list[tuple[float, float]],
    intersection_centers: list[tuple[float, float]],
    crop_w: int,
    crop_h: int,
    stone_radii: list[float] | None = None,
) -> FusedLattice:
    """Fit pitch + origin per axis from segments and point detections.

    If `stone_radii` is given, the median radius sets a hard pitch
    floor (PITCH_FLOOR_RADIUS_MULT × median_r), which rules out the
    half-pitch ambiguity for any board with a few detected stones.
    """
    horizontals, verticals = _split_segments(segments)
    seg_xs = [0.5 * (s.x1 + s.x2) for s in verticals]
    seg_ys = [0.5 * (s.y1 + s.y2) for s in horizontals]

    pt_xs = [p[0] for p in stone_centers] + [p[0] for p in intersection_centers]
    pt_ys = [p[1] for p in stone_centers] + [p[1] for p in intersection_centers]

    pitch_floor = PITCH_MIN
    if stone_radii:
        pitch_floor = max(pitch_floor,
                          PITCH_FLOOR_RADIUS_MULT * float(np.median(stone_radii)))

    x_axis = _fit_axis(seg_xs + pt_xs, pitch_floor=pitch_floor)
    y_axis = _fit_axis(seg_ys + pt_ys, pitch_floor=pitch_floor)
    return FusedLattice(x=x_axis, y=y_axis)


def _split_segments(
    segments: list[Segment],
) -> tuple[list[Segment], list[Segment]]:
    horizontals: list[Segment] = []
    verticals: list[Segment] = []
    for s in segments:
        ang = abs(np.degrees(np.arctan2(s.y2 - s.y1, s.x2 - s.x1))) % 180.0
        if ang > 90.0:
            ang = 180.0 - ang
        if ang <= ORIENT_TOLERANCE_DEG:
            horizontals.append(s)
        elif ang >= 90.0 - ORIENT_TOLERANCE_DEG:
            verticals.append(s)
    return horizontals, verticals


def _fit_axis(positions: list[float], pitch_floor: float = PITCH_MIN) -> AxisLattice:
    if len(positions) < MIN_POSITIONS:
        return AxisLattice(pitch=None, origin=None)
    coords = np.asarray(positions, dtype=float)
    candidates = [p for p in _candidate_pitches(coords) if p >= pitch_floor]
    if not candidates:
        return AxisLattice(pitch=None, origin=None)

    best_count = 0
    best_pitch: float | None = None
    best_origin: float | None = None
    for p in candidates:
        count, origin = _score_pitch(coords, p)
        if count > best_count:
            best_count = count
            best_pitch = p
            best_origin = origin
    if best_pitch is None or best_count < MIN_INLIERS:
        return AxisLattice(pitch=None, origin=None)

    pitch, origin = _refine(coords, best_pitch, best_origin)
    # Normalize origin to the smallest occupied integer lattice line so
    # the returned origin is the page-pixel position of col_local=0.
    cols = np.round((coords - origin) / pitch).astype(int)
    inlier_mask = np.abs(coords - (origin + cols * pitch)) < SNAP_TOL_FRAC * pitch
    if inlier_mask.any():
        col_min = int(cols[inlier_mask].min())
        origin = origin + col_min * pitch
    return AxisLattice(pitch=float(pitch), origin=float(origin))


def _candidate_pitches(coords: np.ndarray) -> list[float]:
    """Smoothed pairwise-diff histogram → local maxima (peak-height-ranked)."""
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
    smoothed = np.convolve(counts, _SMOOTH_KERNEL, mode="same")
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((float(smoothed[i]), float(bins[i]) + 0.5))
    peaks.sort(reverse=True)
    return [p for _, p in peaks]


def _score_pitch(coords: np.ndarray, pitch: float) -> tuple[int, float]:
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
    cols = np.round((coords - origin) / pitch).astype(int)
    residuals = np.abs(coords - (origin + cols * pitch))
    inlier_mask = residuals < SNAP_TOL_FRAC * pitch
    if inlier_mask.sum() < MIN_INLIERS:
        return pitch, origin
    x_in = coords[inlier_mask]
    c_in = cols[inlier_mask].astype(float)
    A = np.vstack([c_in, np.ones(len(c_in))]).T
    p_ref, o_ref = np.linalg.lstsq(A, x_in, rcond=None)[0]
    if PITCH_MIN <= p_ref <= PITCH_MAX:
        return float(p_ref), float(o_ref)
    return pitch, origin
