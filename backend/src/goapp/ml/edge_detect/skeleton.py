"""Skeleton + per-stone edge detector — the new edge classifier.

Wraps the per-piece logic into one entry point so both the dev-tool
endpoint and the discretization pipeline use the same edge decisions.

Pipeline per board crop:
  1. Find the main grid CC bbox (used to clip the per-stone half-plane
     ink check, so captions outside the board don't count).
  2. Paint stones out of the crop, skeletonize the result, recover
     T/L/+ junctions.
  3. Drop junctions whose centroid sits inside any painted stone disc
     — paint-boundary artifacts that look like real L's but aren't.
  4. Re-tally per-side T/L cluster votes from the filtered junctions.
  5. Per-stone edge classification on the **raw** crop: a stone is on
     edge D iff no neighbor stone within ~1 pitch in D AND no grid ink
     in the half-plane past the stone in D (clipped to the grid bbox).
  6. Edge fires iff the cluster vote passes OR ≥2 stones independently
     classified as edge-touching on that side.
  7. Sanity: reject any fired edge if a stone center sits more than
     ~1 stone radius past the asserted edge position in the outward
     direction, OR if a grid-like pattern (≥2 parallel lines at the
     board's pitch) extends past the rim — that's a windowed view, not
     a real edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..stone_detect.clean import paint_out_stones, paint_radius
from ..stone_detect.edge_test import PITCH_FROM_R, StoneEdgeClass, classify_stone_edges
from .tjunction import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    EDGE_COLINEAR_FRAC,
    Junction,
    SideTally,
    detect_junctions,
    main_grid_bbox,
    tally_edges,
)


# A side fires from stones-only votes only if at least this many stones
# classified as edge-touching on that side. Real edges with stones on
# them tend to also generate T-junctions along the same line; a solo
# stone vote with zero T's is more often a misfire than a real edge.
MIN_STONE_VOTES = 2

# A candidate edge fires only if the outward region is NOT a windowed
# view. The signal that we're at the real edge of the board (not a
# windowed view) is that there's no grid going out past the rim — i.e.,
# nothing is on the other side of it. A windowed view, by contrast,
# shows multiple parallel grid lines (or tick marks at pitch spacing)
# continuing past the visible last row. A single isolated line — a
# publisher's page-frame, a binding shadow, a caption rule — does NOT
# count as continuation; only a periodic grid-like pattern at the
# board's pitch does.
OUTWARD_DEPTH_PITCH = 2.5         # scan this many pitches past the rim
OUTWARD_LINE_FRAC = 0.5           # peak: ink ≥ this fraction of band depth
OUTWARD_PITCH_TOL_FRAC = 0.35     # gap-to-pitch tolerance

_SIDE_TO_DIR = {"left": "W", "right": "E", "top": "N", "bottom": "S"}


@dataclass(frozen=True)
class SkeletonEdgeResult:
    edges: dict[str, bool]
    junctions: list[Junction]   # painted-disc artifacts already filtered
    sides: dict[str, SideTally]  # tallies from the filtered junctions
    stone_edges: list[StoneEdgeClass]


def decide_edges(
    crop_bgr: np.ndarray,
    stones: list[dict],
) -> SkeletonEdgeResult:
    h, w = crop_bgr.shape[:2]
    grid_bbox = main_grid_bbox(crop_bgr)

    cleaned = paint_out_stones(crop_bgr, stones)
    raw_junctions = detect_junctions(cleaned).junctions
    stone_edges = classify_stone_edges(crop_bgr, stones, grid_bbox=grid_bbox)

    painted = [(s["x"], s["y"], paint_radius(s["r"])) for s in stones]
    junctions = [j for j in raw_junctions if not _inside_any(j.x, j.y, painted)]

    _, cluster_edges = tally_edges(junctions, w, h)

    edges = {
        side: bool(
            cluster_edges[side]
            or sum(1 for se in stone_edges if se.sides[_SIDE_TO_DIR[side]])
            >= MIN_STONE_VOTES
        )
        for side in ("left", "right", "top", "bottom")
    }

    if stones:
        median_r = float(np.median([s["r"] for s in stones]))
    else:
        median_r = 0.0
    margin = max(median_r, 1.0)
    pitch = median_r * PITCH_FROM_R if median_r > 0 else 0.0

    # Pre-compute the binary used for outward-content scanning. Same
    # adaptive-threshold params as the skeletonizer, so what we count
    # as "ink past the edge" matches what produced the junctions.
    bi = _binarize(crop_bgr)
    for side in ("left", "right", "top", "bottom"):
        if not edges[side]:
            continue
        d = _SIDE_TO_DIR[side]
        ep = _edge_position(d, junctions, stone_edges, crop_h=h, crop_w=w)
        if ep is None:
            continue
        if _stone_beyond(d, ep, stones, margin):
            edges[side] = False
            continue
        # Windowed-view detection: a real edge has nothing past it; a
        # windowed view shows perpendicular grid lines (or tick marks)
        # extending past the visible last row at pitch spacing. Only
        # the latter rejects the edge.
        if _outward_has_grid(bi, side, ep, grid_bbox, margin, pitch):
            edges[side] = False

    sides, _ = tally_edges(junctions, w, h)
    return SkeletonEdgeResult(
        edges=edges, junctions=junctions, sides=sides,
        stone_edges=stone_edges,
    )


def _inside_any(x: float, y: float, discs: list[tuple[float, float, int]]) -> bool:
    for px, py, pr in discs:
        if (x - px) ** 2 + (y - py) ** 2 <= pr * pr:
            return True
    return False


def _edge_position(
    d: str,
    junctions: list[Junction],
    stone_edges: list[StoneEdgeClass],
    crop_h: int | None = None,
    crop_w: int | None = None,
) -> float | None:
    voting_j = [
        j for j in junctions
        if j.kind in ("T", "L") and d in j.outward
    ]
    if voting_j:
        # Stone paint-out can sever interior intersections into phantom
        # T's that vote for the wrong side. Take the median of the
        # largest co-linear cluster (matches tally_edges' notion of a
        # firing edge) instead of the median of all voters.
        coords = sorted(j.y if d in ("N", "S") else j.x for j in voting_j)
        if d in ("N", "S") and crop_h is not None:
            tol = EDGE_COLINEAR_FRAC * crop_h
        elif d in ("E", "W") and crop_w is not None:
            tol = EDGE_COLINEAR_FRAC * crop_w
        else:
            tol = 0.0
        cluster = _largest_cluster(coords, tol) if tol > 0 else coords
        return cluster[len(cluster) // 2]
    voting_s = [se for se in stone_edges if se.sides[d]]
    if not voting_s:
        return None
    if d == "N":
        return min(s.y for s in voting_s)
    if d == "S":
        return max(s.y for s in voting_s)
    if d == "W":
        return min(s.x for s in voting_s)
    return max(s.x for s in voting_s)


def _largest_cluster(sorted_pos: list[float], tol: float) -> list[float]:
    """Largest sliding-window of sorted positions whose span ≤ tol."""
    if not sorted_pos:
        return []
    best = (0, 0)  # (i, j) inclusive
    j = 0
    for i in range(len(sorted_pos)):
        while sorted_pos[i] - sorted_pos[j] > tol:
            j += 1
        if i - j > best[1] - best[0]:
            best = (j, i)
    return sorted_pos[best[0]:best[1] + 1]


def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if crop_bgr.ndim == 3 else crop_bgr
    )
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )


def _outward_has_grid(
    bi: np.ndarray,
    side: str,
    edge_pos: float,
    grid_bbox: tuple[int, int, int, int] | None,
    margin: float,
    pitch: float,
) -> bool:
    """True iff a grid-like pattern (≥2 parallel lines spaced at the
    board's pitch) extends past `edge_pos` outward. This is the only
    accepted "board continues past this view" signal: a single isolated
    line — page-frame rule, binding shadow, caption rule — does NOT
    qualify, and the edge stays fired.

    Projects the outward band onto the axis PERPENDICULAR to the rim:
    for a horizontal rim (top/bottom), continuation = vertical lines
    extending past it, which appear as periodic peaks along x.
    """
    if pitch <= 0:
        return False
    H, W = bi.shape
    if grid_bbox is None:
        gx0, gy0, gx1, gy1 = 0, 0, W, H
    else:
        gx0, gy0, gx1, gy1 = grid_bbox
    skip = int(max(margin, 8))
    depth = int(max(OUTWARD_DEPTH_PITCH * pitch, 30))
    if side == "top":
        y_hi = max(0, int(edge_pos) - skip)
        y_lo = max(0, y_hi - depth)
        band = bi[y_lo:y_hi, gx0:gx1]
        proj = (band > 0).sum(axis=0)
        actual_depth = band.shape[0]
    elif side == "bottom":
        y_lo = min(H, int(edge_pos) + skip)
        y_hi = min(H, y_lo + depth)
        band = bi[y_lo:y_hi, gx0:gx1]
        proj = (band > 0).sum(axis=0)
        actual_depth = band.shape[0]
    elif side == "left":
        x_hi = max(0, int(edge_pos) - skip)
        x_lo = max(0, x_hi - depth)
        band = bi[gy0:gy1, x_lo:x_hi]
        proj = (band > 0).sum(axis=1)
        actual_depth = band.shape[1]
    else:  # right
        x_lo = min(W, int(edge_pos) + skip)
        x_hi = min(W, x_lo + depth)
        band = bi[gy0:gy1, x_lo:x_hi]
        proj = (band > 0).sum(axis=1)
        actual_depth = band.shape[1]
    if proj.size == 0 or actual_depth < 2 * pitch:
        # Not enough room past the rim to ever fit two parallel lines.
        return False
    peak_thresh = max(3, int(OUTWARD_LINE_FRAC * actual_depth))
    occ = (proj >= peak_thresh).tolist()
    lines = _contiguous_clusters(occ)
    if len(lines) < 2:
        return False
    centers = [(s + e) / 2.0 for s, e in lines]
    tol = OUTWARD_PITCH_TOL_FRAC * pitch
    return any(
        abs((centers[i + 1] - centers[i]) - pitch) <= tol
        for i in range(len(centers) - 1)
    )


def _contiguous_clusters(mask: list[bool]) -> list[tuple[int, int]]:
    """Return inclusive (start, end) ranges for each run of True."""
    out: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def _stone_beyond(
    d: str, edge_pos: float, stones: list[dict], margin: float,
) -> bool:
    for s in stones:
        if d == "N" and s["y"] < edge_pos - margin:
            return True
        if d == "S" and s["y"] > edge_pos + margin:
            return True
        if d == "W" and s["x"] < edge_pos - margin:
            return True
        if d == "E" and s["x"] > edge_pos + margin:
            return True
    return False
