"""Classical grid fit: given detected stone pixel positions + an initial
(pitch, origin) estimate, iteratively refine the grid geometry so that
stones line up with integer intersections. No training, no gradients —
just assign-to-nearest-intersection + linear least squares, looped.

Unlike the snap regressor, this uses the stone heatmap peaks as evidence
for where the grid actually is. Many stones = strong overdetermined fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FitResult:
    pitch_x_px: float
    pitch_y_px: float
    origin_x_px: float
    origin_y_px: float
    inliers: int
    iterations: int


def fit_grid_to_stones(
    stones: Iterable[dict],            # each has "x", "y" in pixels in the original crop
    pitch_x_init: float,
    pitch_y_init: float,
    origin_x_init: float,
    origin_y_init: float,
    col_min: int,
    col_max: int,
    row_min: int,
    row_max: int,
    iters: int = 6,
    inlier_tol: float = 0.35,          # fraction of pitch; residuals within this count as inliers
) -> FitResult:
    """Iteratively refine (pitch_x, pitch_y, origin_x, origin_y) in pixel space
    using stone positions as evidence.

    Each iteration:
      1. Assign every stone to its nearest grid intersection within the window.
      2. Drop stones whose per-axis residual exceeds `inlier_tol * pitch`.
      3. Linear least-squares on the inliers: x_i = origin_x + (c_i - col_min) * pitch_x.
    """
    px = max(1.0, pitch_x_init)
    py = max(1.0, pitch_y_init)
    ox = origin_x_init
    oy = origin_y_init

    xs = np.array([s["x"] for s in stones], dtype=np.float64)
    ys = np.array([s["y"] for s in stones], dtype=np.float64)
    if xs.size < 3:
        return FitResult(px, py, ox, oy, inliers=int(xs.size), iterations=0)

    last_inliers = 0
    it = 0
    for it in range(1, iters + 1):
        cx = np.clip(np.round((xs - ox) / px).astype(int) + col_min, col_min, col_max)
        cy = np.clip(np.round((ys - oy) / py).astype(int) + row_min, row_min, row_max)
        tx = ox + (cx - col_min) * px
        ty = oy + (cy - row_min) * py
        inlier = (np.abs(xs - tx) < inlier_tol * px) & (np.abs(ys - ty) < inlier_tol * py)
        n_in = int(inlier.sum())
        if n_in < 3:
            break

        A_x = np.column_stack([np.ones(n_in), (cx[inlier] - col_min).astype(float)])
        ox_new, px_new = np.linalg.lstsq(A_x, xs[inlier], rcond=None)[0]
        A_y = np.column_stack([np.ones(n_in), (cy[inlier] - row_min).astype(float)])
        oy_new, py_new = np.linalg.lstsq(A_y, ys[inlier], rcond=None)[0]

        if (abs(ox_new - ox) < 0.1 and abs(oy_new - oy) < 0.1
                and abs(px_new - px) < 0.05 and abs(py_new - py) < 0.05
                and n_in == last_inliers):
            ox, oy, px, py = ox_new, oy_new, px_new, py_new
            break

        ox, oy, px, py = ox_new, oy_new, px_new, py_new
        last_inliers = n_in

    # Sanity: refuse to return a fit with absurd pitch.
    if px < 1.0 or py < 1.0:
        return FitResult(pitch_x_init, pitch_y_init, origin_x_init, origin_y_init,
                         inliers=0, iterations=it)

    return FitResult(
        pitch_x_px=float(px), pitch_y_px=float(py),
        origin_x_px=float(ox), origin_y_px=float(oy),
        inliers=last_inliers, iterations=it,
    )
