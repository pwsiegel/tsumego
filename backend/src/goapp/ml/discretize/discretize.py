"""Stone-pixel-positions → discrete 19x19 grid assignment.

Classical, no training. Given a list of stone centers (from the stone CNN)
plus the 4-bit edge classifier output, figure out:
  - cell_size (pitch in crop pixels)
  - origin_x, origin_y (pixel position of the top-left visible intersection)
  - visible_cols / visible_rows (grid extent inside the crop)
  - for each stone, its (col, row) on the 19x19 board
  - placement of the visible window on the full 19x19

Approach:
  - cell_size: 25th percentile of pairwise axis-aligned stone distances
    within a plausible pitch range. The lower quartile is dominated by
    nearest-neighbor (adjacent-cell) distances.
  - origin: brute-force search in [0, cell_size) to minimize snap residual.
  - window placement: edges["left"|"right"|"top"|"bottom"] from the edge
    classifier disambiguate which region of the 19x19 the crop shows.
    When an axis has no real-boundary edge, fall back to centering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOARD_SIZE = 19


@dataclass(frozen=True)
class DiscretizedStone:
    x: float        # pixel position in crop
    y: float
    color: str      # "B" or "W"
    conf: float
    col_local: int  # 0-indexed from the top-left visible intersection
    row_local: int
    col: int        # absolute column on 19x19 (col_min + col_local)
    row: int        # absolute row on 19x19


@dataclass(frozen=True)
class Discretized:
    cell_size: float
    origin_x: float
    origin_y: float
    visible_cols: int
    visible_rows: int
    col_min: int
    row_min: int
    stones: list[DiscretizedStone]


def discretize(
    stones: list[dict],         # each: {"x", "y", "color", "conf"}
    crop_w: int,
    crop_h: int,
    edges: dict[str, bool] | None = None,
    conf_thresh: float = 0.0,
    cell_size_override: float | None = None,
    origin_x_override: float | None = None,
    origin_y_override: float | None = None,
    pitch_x_override: float | None = None,
    pitch_y_override: float | None = None,
) -> Discretized:
    edges = edges or {"left": False, "right": False, "top": False, "bottom": False}
    usable = [s for s in stones if float(s.get("conf", 1.0)) >= conf_thresh]
    if len(usable) < 2:
        return _empty(crop_w, crop_h)

    xs = np.array([float(s["x"]) for s in usable])
    ys = np.array([float(s["y"]) for s in usable])

    if cell_size_override is not None and cell_size_override > 0:
        cell_size = float(cell_size_override)
    else:
        cell_size = _estimate_cell_size(xs, ys, crop_w, crop_h)
    if cell_size <= 0:
        return _empty(crop_w, crop_h)
    # Per-axis pitch (handles scans stretched unequally in x/y). Falls
    # back to the single cell_size when axis-specific pitch isn't known.
    pitch_x = float(pitch_x_override) if pitch_x_override is not None and pitch_x_override > 0 else cell_size
    pitch_y = float(pitch_y_override) if pitch_y_override is not None and pitch_y_override > 0 else cell_size
    origin_x = (
        float(origin_x_override) if origin_x_override is not None
        else _estimate_origin_1d(xs, pitch_x)
    )
    origin_y = (
        float(origin_y_override) if origin_y_override is not None
        else _estimate_origin_1d(ys, pitch_y)
    )

    visible_cols = max(
        1, int(np.floor((crop_w - 1 - origin_x) / pitch_x)) + 1
    )
    visible_rows = max(
        1, int(np.floor((crop_h - 1 - origin_y) / pitch_y)) + 1
    )
    visible_cols = min(BOARD_SIZE, visible_cols)
    visible_rows = min(BOARD_SIZE, visible_rows)

    col_min, row_min = _place_window(edges, visible_cols, visible_rows)

    # Snap each detection; then dedupe by (col, row) keeping the
    # highest-confidence stone. Two detections with different colors
    # can land on the same intersection (e.g. a dark-rim-around-a-light-
    # interior ambiguous pattern triggers a B and W detection very close
    # in pixel space that both snap to the same cell). Downstream rendering
    # and SGF emission rely on cell positions being unique.
    by_cell: dict[tuple[int, int], DiscretizedStone] = {}
    for s, x, y in zip(usable, xs, ys):
        c_local = int(round((float(x) - origin_x) / pitch_x))
        r_local = int(round((float(y) - origin_y) / pitch_y))
        c_local = max(0, min(visible_cols - 1, c_local))
        r_local = max(0, min(visible_rows - 1, r_local))
        entry = DiscretizedStone(
            x=float(x), y=float(y),
            color=str(s["color"]),
            conf=float(s.get("conf", 1.0)),
            col_local=c_local, row_local=r_local,
            col=col_min + c_local, row=row_min + r_local,
        )
        key = (entry.col, entry.row)
        prev = by_cell.get(key)
        if prev is None or entry.conf > prev.conf:
            by_cell[key] = entry
    snapped = list(by_cell.values())

    return Discretized(
        cell_size=float(cell_size),
        origin_x=float(origin_x), origin_y=float(origin_y),
        visible_cols=visible_cols, visible_rows=visible_rows,
        col_min=col_min, row_min=row_min,
        stones=snapped,
    )


def _estimate_cell_size(
    xs: np.ndarray, ys: np.ndarray,
    crop_w: int, crop_h: int,
) -> float:
    min_pitch = max(8.0, min(crop_w, crop_h) / 22.0)
    max_pitch = max(crop_w, crop_h) / 3.5
    N = len(xs)
    if N < 2:
        return 0.0
    ones_x = np.abs(xs[:, None] - xs[None, :])
    ones_y = np.abs(ys[:, None] - ys[None, :])
    # Upper-triangular mask so we don't double-count pairs.
    mask = np.triu(np.ones_like(ones_x, dtype=bool), k=1)
    dists = np.concatenate([ones_x[mask], ones_y[mask]])
    plausible = dists[(dists >= min_pitch) & (dists <= max_pitch)]
    if plausible.size == 0:
        return 0.0
    # Lowest-percentile of plausible 1D distances is the cell pitch:
    # adjacent-cell spacings cluster at the bottom of the distribution
    # even when stones are sparse or clumped (fewer adjacent pairs than
    # multi-cell pairs). 25th percentile is right for densely-populated
    # boards but overshoots 2× on sparse ones; 5th is robust across both.
    return float(np.percentile(plausible, 5))


def _estimate_origin_1d(vals: np.ndarray, cell_size: float) -> float:
    """Find origin in [0, cell_size) that minimizes snap residual."""
    if vals.size == 0 or cell_size <= 0:
        return 0.0
    best_origin = 0.0
    best_cost = float("inf")
    # 100-step brute force is ~cell_size/100 ≈ 0.3 px precision; good enough.
    for o in np.linspace(0.0, cell_size, 101, endpoint=False):
        offset = vals - o
        snapped = offset - np.round(offset / cell_size) * cell_size
        cost = float(np.sum(snapped * snapped))
        if cost < best_cost:
            best_cost = cost
            best_origin = float(o)
    return best_origin


def _place_window(
    edges: dict[str, bool],
    visible_cols: int,
    visible_rows: int,
) -> tuple[int, int]:
    """Decide (col_min, row_min) on the 19x19 from edge bits.

    left+top → top-left corner, col_min=row_min=0.
    left+bottom → bottom-left, col_min=0, row_min=18-(visible_rows-1).
    Pure middle (no real edges) centers the window.
    """
    if edges.get("left", False):
        col_min = 0
    elif edges.get("right", False):
        col_min = BOARD_SIZE - visible_cols
    else:
        col_min = max(0, (BOARD_SIZE - visible_cols) // 2)
    if edges.get("top", False):
        row_min = 0
    elif edges.get("bottom", False):
        row_min = BOARD_SIZE - visible_rows
    else:
        row_min = max(0, (BOARD_SIZE - visible_rows) // 2)
    col_min = max(0, min(BOARD_SIZE - visible_cols, col_min))
    row_min = max(0, min(BOARD_SIZE - visible_rows, row_min))
    return col_min, row_min


def _empty(crop_w: int, crop_h: int) -> Discretized:
    return Discretized(
        cell_size=0.0, origin_x=0.0, origin_y=0.0,
        visible_cols=0, visible_rows=0,
        col_min=0, row_min=0,
        stones=[],
    )
