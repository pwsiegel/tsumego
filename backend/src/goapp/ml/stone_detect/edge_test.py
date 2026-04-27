"""Per-stone, per-cardinal "is this stone on the board edge?" test.

For each detected stone and each of N/E/S/W, decide whether the stone
sits on that edge of the board using two purely local signals on the
**raw** crop (not the painted-out one — the painting can erase the very
edge we're looking for):

  1. No neighbor stone within ~1 pitch in that direction.
  2. No grid ink in the half-plane extending from the stone outward in
     that direction, clipped to the main-grid CC bbox so captions and
     page numbers below the board don't count.

Both must hold. If either fails — there's a stone next door, or there's
ink past the stone in that direction — the stone is not on that edge.

The "no ink in the whole half-plane" formulation is much stronger than
checking a thin strip at the predicted next intersection: a true edge
stone has nothing beyond it; an interior stone has many grid lines
beyond it that are obvious in aggregate even if any single line is
faint or slightly mis-aligned with a pitch estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Direction = Literal["N", "E", "S", "W"]
_DIRS: tuple[Direction, ...] = ("N", "E", "S", "W")
_DIR_VEC: dict[Direction, tuple[int, int]] = {
    "N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0),
}

# Stone radius from the YOLO detector measures the heatmap-peak "core",
# noticeably smaller than the visible outer outline. Pitch in real PDFs
# typically runs ~2.4–2.6× this core radius.
PITCH_FROM_R = 2.5

# Match the tjunction skeleton binarization so the ink test sees the
# same lines the edge detector reasons about.
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8

# A neighbor stone counts if its center sits within
# NEIGHBOR_RANGE_PITCH × pitch along the test direction and within
# NEIGHBOR_PERP_PITCH × pitch perpendicular to it.
NEIGHBOR_RANGE_PITCH = 1.5
NEIGHBOR_PERP_PITCH = 0.5

# Ink-strip geometry. The strip starts past the stone's visible outline
# (so its own ring isn't counted as ink) and extends to the main-grid
# bbox edge in the test direction. Perpendicular half-width scales with
# stone radius — wide enough to catch a slightly skewed grid line
# crossing the stone's row/column, narrow enough to ignore neighboring
# rows/columns.
STRIP_START_R = 1.4
STRIP_HALF_WIDTH_R = 1.2

# Projection-based ink test: sum the binary strip along the
# perpendicular axis (rows for N/S, columns for E/W). A grid line
# appears as a single row/column whose count is close to the full
# perpendicular strip dimension (line length). We declare "ink
# present" if any perpendicular slice of the strip has at least this
# fraction of its pixels above threshold.
#
# Why projection instead of total pixel count: when a stone sits one
# row off the edge, the half-plane strip between it and the edge is
# tiny — a real edge line in there might be only ~24 black pixels,
# below any reasonable absolute count threshold. But projected onto
# rows, that line still occupies a single row with ~12 of 12 pixels
# black — easily distinguishable from noise.
INK_LINE_FRAC = 0.4



@dataclass(frozen=True)
class StoneEdgeClass:
    x: float
    y: float
    r: float
    color: str  # "B" | "W"
    sides: dict[Direction, bool]  # which cardinals are "edge"


def classify_stone_edges(
    crop_bgr: np.ndarray,
    stones: list[dict],
    grid_bbox: tuple[int, int, int, int] | None = None,
    pitch: float | None = None,
) -> list[StoneEdgeClass]:
    """Per-stone edge classification on the raw crop.

    `stones` entries follow the stone_detect.detect schema: dicts with
    keys x, y, r, color, conf. `grid_bbox` is the (x0,y0,x1,y1) bbox of
    the main board CC (from edge_detect.tjunction.main_grid_bbox);
    used to clip the ink search so captions outside the grid don't
    count. Falls back to the full crop bbox if not provided. `pitch` is
    used only for the neighbor-stone proximity check; estimated from
    radius if omitted.
    """
    if not stones:
        return []
    if pitch is None:
        pitch = float(np.median([s["r"] for s in stones])) * PITCH_FROM_R

    binary = _binarize(crop_bgr)
    H, W = binary.shape
    if grid_bbox is None:
        grid_bbox = (0, 0, W, H)

    centers = np.array([(s["x"], s["y"]) for s in stones], dtype=np.float32)

    out: list[StoneEdgeClass] = []
    for i, s in enumerate(stones):
        cx = float(s["x"])
        cy = float(s["y"])
        r = float(s["r"])
        sides: dict[Direction, bool] = {}
        for d in _DIRS:
            sides[d] = _is_edge(
                cx=cx, cy=cy, r=r, pitch=pitch, direction=d,
                self_idx=i, centers=centers, binary=binary,
                grid_bbox=grid_bbox,
            )
        out.append(
            StoneEdgeClass(
                x=cx, y=cy, r=r, color=s["color"], sides=sides,
            )
        )
    return out


def _is_edge(
    *, cx: float, cy: float, r: float, pitch: float,
    direction: Direction, self_idx: int, centers: np.ndarray,
    binary: np.ndarray, grid_bbox: tuple[int, int, int, int],
) -> bool:
    if _has_neighbor(cx, cy, pitch, direction, self_idx, centers):
        return False
    return not _has_ink(cx, cy, r, direction, binary, grid_bbox)


def _has_neighbor(
    cx: float, cy: float, pitch: float, direction: Direction,
    self_idx: int, centers: np.ndarray,
) -> bool:
    dx, dy = _DIR_VEC[direction]
    rel = centers - np.array([cx, cy], dtype=np.float32)
    along = rel[:, 0] * dx + rel[:, 1] * dy
    perp = rel[:, 0] * (-dy) + rel[:, 1] * dx
    along_ok = (along > 0.1 * pitch) & (along < NEIGHBOR_RANGE_PITCH * pitch)
    perp_ok = np.abs(perp) < NEIGHBOR_PERP_PITCH * pitch
    hits = along_ok & perp_ok
    hits[self_idx] = False
    return bool(hits.any())


def _has_ink(
    cx: float, cy: float, r: float, direction: Direction,
    binary: np.ndarray, grid_bbox: tuple[int, int, int, int],
) -> bool:
    """Strip from past the stone's outline to the grid bbox edge in the
    test direction; perpendicular half-width tied to stone radius. Any
    significant black-pixel count → grid continues past the stone."""
    near = STRIP_START_R * r
    half_w = STRIP_HALF_WIDTH_R * r
    gx0, gy0, gx1, gy1 = grid_bbox

    if direction == "W":
        xa, xb = gx0, cx - near
        ya, yb = cy - half_w, cy + half_w
    elif direction == "E":
        xa, xb = cx + near, gx1
        ya, yb = cy - half_w, cy + half_w
    elif direction == "N":
        xa, xb = cx - half_w, cx + half_w
        ya, yb = gy0, cy - near
    else:  # "S"
        xa, xb = cx - half_w, cx + half_w
        ya, yb = cy + near, gy1

    H, W = binary.shape
    xa_i = max(0, int(round(xa)))
    xb_i = min(W, int(round(xb)))
    ya_i = max(0, int(round(ya)))
    yb_i = min(H, int(round(yb)))
    if xa_i >= xb_i or ya_i >= yb_i:
        return False  # nothing to look at past the stone → treat as no ink

    region = binary[ya_i:yb_i, xa_i:xb_i] > 0
    if region.size == 0:
        return False
    if direction in ("N", "S"):
        # Sum over columns (axis=1): peak = brightest row in the strip.
        peak = int(region.sum(axis=1).max())
        full = region.shape[1]
    else:
        # Sum over rows (axis=0): peak = brightest column in the strip.
        peak = int(region.sum(axis=0).max())
        full = region.shape[0]
    if full == 0:
        return False
    return (peak / full) >= INK_LINE_FRAC


def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3
        else crop_bgr
    )
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )
