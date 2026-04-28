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
     direction, OR if ≥3 grid junctions sit past the asserted edge —
     that's a windowed view (the previous problem's diagram, or more
     of the same board), not a real edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stone_detect.clean import paint_out_stones, paint_radius
from ..stone_detect.edge_test import StoneEdgeClass, classify_stone_edges
from .tjunction import (
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
# view. The signal we use: vertical/horizontal STUBS extending out of
# the rim — short line segments on the 1-px skeleton, sticking up (or
# sideways) from each grid intersection. A real edge has no stubs past
# it; a windowed view has stubs at every grid column past it (because
# the board's grid lines continue through the asserted "edge"). We
# count stubs on the skeleton rather than ink in the binarization
# because the skeleton is robust against faint photocopies, dirty
# scans, and answer-stone blobs that defeat raw thresholding.
OUTWARD_STUBS_THRESH = 3            # ≥ this many stubs past the rim → not a real edge
OUTWARD_STUB_MIN_LEN = 10           # a stub must run this many px into the band to count;
                                    # short leakage from the grid CC bounds is ignored
OUTWARD_STUB_MAX_DEPTH = 40         # scan this far past the rim, in pixels

_SIDE_TO_DIR = {"left": "W", "right": "E", "top": "N", "bottom": "S"}


@dataclass(frozen=True)
class SkeletonEdgeResult:
    edges: dict[str, bool]
    # Pixel position of each detected edge: x for "left"/"right", y for
    # "top"/"bottom". None when the side wasn't accepted as a real edge.
    # Lets downstream lattice fitting hard-anchor origin/pitch instead of
    # relying on a 1D phase search that can land off-by-one on curved scans.
    edge_positions: dict[str, float | None]
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
    tj = detect_junctions(cleaned)
    raw_junctions = tj.junctions
    skel = tj.skel
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

    edge_positions: dict[str, float | None] = {
        "left": None, "right": None, "top": None, "bottom": None,
    }
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
        if _outward_has_stubs(skel, side, ep):
            edges[side] = False
            continue
        edge_positions[side] = float(ep)

    sides, _ = tally_edges(junctions, w, h)
    return SkeletonEdgeResult(
        edges=edges, edge_positions=edge_positions,
        junctions=junctions, sides=sides,
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


_OUTWARD_SKIP = 3


def _outward_has_stubs(
    skel: np.ndarray,
    side: str,
    edge_pos: float,
) -> bool:
    """True iff ≥OUTWARD_STUBS_THRESH stubs extend past `edge_pos` on
    the 1-px skeleton. For top/bottom the stubs are vertical (columns
    with skeleton ink); for left/right they're horizontal (rows). A
    column/row counts as a stub if it has ≥OUTWARD_STUB_MIN_LEN skeleton
    pixels in the band past the rim."""
    H, W = skel.shape
    skip = _OUTWARD_SKIP
    depth = OUTWARD_STUB_MAX_DEPTH
    if side == "top":
        y_hi = max(0, int(edge_pos) - skip)
        y_lo = max(0, y_hi - depth)
        band = skel[y_lo:y_hi, :]
        proj = band.sum(axis=0)
    elif side == "bottom":
        y_lo = min(H, int(edge_pos) + skip)
        y_hi = min(H, y_lo + depth)
        band = skel[y_lo:y_hi, :]
        proj = band.sum(axis=0)
    elif side == "left":
        x_hi = max(0, int(edge_pos) - skip)
        x_lo = max(0, x_hi - depth)
        band = skel[:, x_lo:x_hi]
        proj = band.sum(axis=1)
    else:  # right
        x_lo = min(W, int(edge_pos) + skip)
        x_hi = min(W, x_lo + depth)
        band = skel[:, x_lo:x_hi]
        proj = band.sum(axis=1)
    if proj.size == 0:
        return False
    # Each "stub" is a contiguous run of columns/rows above the length
    # threshold — a single thick line shouldn't count more than once.
    occ = proj >= OUTWARD_STUB_MIN_LEN
    stubs = 0
    in_run = False
    for v in occ:
        if v and not in_run:
            stubs += 1
            in_run = True
        elif not v:
            in_run = False
    return stubs >= OUTWARD_STUBS_THRESH


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
