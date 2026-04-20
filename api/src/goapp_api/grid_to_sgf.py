"""Convert a predicted 19x19 grid + window info into an SGF string."""

from __future__ import annotations

import numpy as np


BOARD_SIZE = 19


def grid_to_sgf(grid: np.ndarray, window: dict[str, int]) -> str:
    """grid is (19,19) with values 0=empty, 1=B, 2=W. window tells us which
    cols/rows are actually visible; cells outside are ignored regardless of
    what the model predicted there."""
    c_lo = window["col_min"]; c_hi = window["col_max"]
    r_lo = window["row_min"]; r_hi = window["row_max"]

    def coord(col: int, row: int) -> str:
        return f"{chr(ord('a') + col)}{chr(ord('a') + row)}"

    black: list[tuple[int, int]] = []
    white: list[tuple[int, int]] = []
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            v = int(grid[r, c])
            if v == 1:
                black.append((c, r))
            elif v == 2:
                white.append((c, r))

    parts = [f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]"]
    if black:
        parts.append("AB" + "".join(f"[{coord(c, r)}]" for c, r in black))
    if white:
        parts.append("AW" + "".join(f"[{coord(c, r)}]" for c, r in white))
    parts.append(")")
    return "".join(parts)


def window_from_edges_and_bbox(
    edges: dict[str, bool], bbox_w_cells: int, bbox_h_cells: int,
) -> dict[str, int]:
    """Given which sides are real board boundaries and the rough number of
    cells visible in each direction (derivable from YOLO's tight bbox + the
    model's grid output), infer col_min/col_max/row_min/row_max.

    If both L and R are boundaries → cols [0, 18] (ignore bbox_w_cells).
    Else if only L → cols [0, 0 + bbox_w_cells - 1].
    Else if only R → cols [18 - bbox_w_cells + 1, 18].
    Else (no horizontal boundary) → default to leftmost region [0, bbox_w_cells-1].
    Same logic vertically.
    """
    def axis(low: bool, high: bool, n_cells: int) -> tuple[int, int]:
        if low and high:
            return (0, BOARD_SIZE - 1)
        if low:
            return (0, min(BOARD_SIZE - 1, n_cells - 1))
        if high:
            return (max(0, BOARD_SIZE - n_cells), BOARD_SIZE - 1)
        # Neither: best-effort default to top-left placement.
        return (0, min(BOARD_SIZE - 1, n_cells - 1))

    c_lo, c_hi = axis(edges["left"], edges["right"], bbox_w_cells)
    r_lo, r_hi = axis(edges["top"], edges["bottom"], bbox_h_cells)
    return {"col_min": c_lo, "col_max": c_hi, "row_min": r_lo, "row_max": r_hi}
