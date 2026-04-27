"""Junction detection by skeletonization + neighbor counting.

Approach: binarize the (stones-painted-out) crop, skeletonize down to
1-pixel-wide lines, prune short spurs, then look at each skeleton
pixel's 8-neighborhood. A skeleton pixel with 3 black neighbors is
sitting at a T-junction; 4 neighbors is a +; 1 is an endpoint.

Hough segments lose topology — Hough's voting peaks snap segment
endpoints to nearby intersections, so a column that visually crosses a
horizontal can come back as two segments terminating at the horizontal,
producing phantom T's. Pixel skeletons preserve topology directly:
where lines cross, the skeleton crosses; where one terminates against
another, the skeleton terminates against the other.

Each junction-pixel cluster is collapsed to a centroid, then arm
directions are recovered by walking the skeleton outward a few pixels
from each cluster boundary and snapping the resulting displacement to
N/E/S/W. Side tallies fall out the same way as the segment version.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import convolve, label as nd_label
from skimage.morphology import skeletonize


# Adaptive-threshold params, matching segments.detect — same binarization
# so faint scanned grid lines survive without the outer frame melting.
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8

# Distance to walk along the skeleton from each junction cluster before
# measuring the arm direction. Enough to escape the cluster's local
# fuzziness; short enough that the walk doesn't reach the next junction
# on dense boards.
ARM_WALK_STEPS = 8

# Snap an arm displacement to N/E/S/W if its dominant axis is at least
# this multiple of the perpendicular component. 2.5 ≈ within ~22° of
# axis-aligned — comfortably tolerant of scan skew, strict enough to
# reject diagonals.
ARM_AXIS_RATIO = 2.5

# Cluster pixels with a junction count (≥3 neighbors) are merged into a
# single junction by 8-connected component labeling. Clusters whose
# pixel count exceeds this are discarded as text/noise blobs.
MAX_CLUSTER_PIXELS = 50

# A side fires only when its voting T/L junctions form a co-linear
# group: their perpendicular positions span at most this fraction of
# the crop dimension. Tolerates a couple degrees of scan skew while
# still rejecting scattered phantom T's (e.g. a single T deep inside
# the crop caused by a stone paint-out severing one arm of an interior
# intersection).
EDGE_COLINEAR_FRAC = 0.04

# Minimum number of co-linear T/L junctions on a side before the edge
# fires. Real partial-board edges typically have many; setting to 2
# admits short partial-edge views (e.g. only 2 grid rows of a side
# visible) while still rejecting isolated phantom T's left over after
# stone paint-out severs an interior arm.
MIN_EDGE_CLUSTER = 2

JunctionKind = Literal["T", "L", "+", "I", "?"]
Direction = Literal["N", "E", "S", "W"]

ARM_N, ARM_E, ARM_S, ARM_W = 0b0001, 0b0010, 0b0100, 0b1000


@dataclass(frozen=True)
class Junction:
    x: float
    y: float
    arms: int
    kind: JunctionKind
    outward: list[Direction]


@dataclass(frozen=True)
class SideTally:
    t: int
    l: int
    total: int


@dataclass(frozen=True)
class TJunctionResult:
    junctions: list[Junction]
    sides: dict[str, SideTally]
    edges: dict[str, bool]


def detect_junctions(crop_bgr: np.ndarray) -> TJunctionResult:
    """Skeleton-based junction detection on a stones-painted-out crop."""
    skel = _skeletonize(crop_bgr)

    nbr = _neighbor_count(skel)
    junction_mask = (nbr >= 3) & skel

    # Cluster adjacent junction pixels (8-connectivity).
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n_clusters = nd_label(junction_mask, structure=structure)

    junctions: list[Junction] = []
    for cid in range(1, n_clusters + 1):
        ys, xs = np.where(labels == cid)
        if len(ys) == 0 or len(ys) > MAX_CLUSTER_PIXELS:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())

        arms = _recover_arms(skel, labels, cid, ys, xs)
        if arms == 0:
            continue
        junctions.append(
            Junction(
                x=cx, y=cy, arms=arms,
                kind=_classify(arms), outward=_missing(arms),
            )
        )

    H, W = skel.shape
    sides, edges = tally_edges(junctions, W, H)
    return TJunctionResult(junctions=junctions, sides=sides, edges=edges)


SIDE_DIR: dict[str, Direction] = {
    "left": "W", "right": "E", "top": "N", "bottom": "S",
}


def tally_edges(
    junctions: list[Junction], W: int, H: int,
) -> tuple[dict[str, SideTally], dict[str, bool]]:
    """Count voting T/L's per side and decide whether each edge fires.
    Exposed so callers can re-tally after filtering the junction list
    (e.g. dropping junctions that sit inside painted-out stone discs).
    """
    sides: dict[str, SideTally] = {}
    edges: dict[str, bool] = {}
    for side, d in SIDE_DIR.items():
        voting = [
            j for j in junctions
            if j.kind in ("T", "L") and d in j.outward
        ]
        t = sum(1 for j in voting if j.kind == "T")
        l = sum(1 for j in voting if j.kind == "L")
        sides[side] = SideTally(t=t, l=l, total=t + l)

        if d in ("N", "S"):
            positions = [j.y for j in voting]
            tol = EDGE_COLINEAR_FRAC * H
        else:
            positions = [j.x for j in voting]
            tol = EDGE_COLINEAR_FRAC * W
        edges[side] = _largest_cluster_size(positions, tol) >= MIN_EDGE_CLUSTER
    return sides, edges


def _largest_cluster_size(positions: list[float], tol: float) -> int:
    """Sliding-window cluster size: largest count of sorted positions
    whose max-min spread fits within `tol`. Junctions that share a real
    edge sit close in perpendicular position; scattered phantoms don't."""
    if not positions:
        return 0
    sorted_pos = sorted(positions)
    best = 1
    j = 0
    for i in range(len(sorted_pos)):
        while sorted_pos[i] - sorted_pos[j] > tol:
            j += 1
        if i - j + 1 > best:
            best = i - j + 1
    return best


def main_grid_bbox(crop_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the largest binary CC in the
    crop, which on a board crop is the grid + frame. Used to reject
    captions / page numbers that sit outside that region. Returns None
    if the crop has no foreground.

    Same adaptive-threshold params as `_skeletonize` so callers can
    trust the bbox matches what skeleton-time CC filtering sees."""
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3
        else crop_bgr
    )
    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bi, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    x0 = int(stats[main, cv2.CC_STAT_LEFT])
    y0 = int(stats[main, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[main, cv2.CC_STAT_WIDTH])
    y1 = y0 + int(stats[main, cv2.CC_STAT_HEIGHT])
    return x0, y0, x1, y1


def _skeletonize(crop_bgr: np.ndarray) -> np.ndarray:
    """Adaptive-threshold, drop ink components that sit outside the
    main grid's bounding box (captions, page numbers), then thin to a
    1-px skeleton. Returns bool array.

    Why bbox-overlap and not just largest-CC: stone paint-out can sever
    parts of the grid (e.g. the leftmost column when a stone sits on
    it) into smaller components. Largest-CC alone drops those grid
    fragments along with the captions; keeping any CC whose bbox
    overlaps the main grid's bbox preserves the fragments while still
    rejecting text that sits outside the grid region."""
    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3
        else crop_bgr
    )
    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bi, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main = 1 + int(np.argmax(areas))
        mx0 = int(stats[main, cv2.CC_STAT_LEFT])
        my0 = int(stats[main, cv2.CC_STAT_TOP])
        mx1 = mx0 + int(stats[main, cv2.CC_STAT_WIDTH])
        my1 = my0 + int(stats[main, cv2.CC_STAT_HEIGHT])

        keep_labels: list[int] = []
        for i in range(1, n_labels):
            cx0 = int(stats[i, cv2.CC_STAT_LEFT])
            cy0 = int(stats[i, cv2.CC_STAT_TOP])
            cx1 = cx0 + int(stats[i, cv2.CC_STAT_WIDTH])
            cy1 = cy0 + int(stats[i, cv2.CC_STAT_HEIGHT])
            if not (cx1 < mx0 or cx0 > mx1 or cy1 < my0 or cy0 > my1):
                keep_labels.append(i)
        keep_mask = np.isin(labels, keep_labels)
        bi = np.where(keep_mask, 255, 0).astype(np.uint8)
    skel = skeletonize(bi > 0)
    return skel.astype(bool)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """8-neighbor count for each True pixel; 0 elsewhere."""
    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32,
    )
    counts = convolve(skel.astype(np.int32), kernel, mode="constant", cval=0)
    counts[~skel] = 0
    return counts


def _recover_arms(
    skel: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    ys: np.ndarray,
    xs: np.ndarray,
) -> int:
    """For each skeleton pixel adjacent to but outside this cluster,
    walk along the skeleton ARM_WALK_STEPS pixels and snap the resulting
    displacement to N/E/S/W. Union of snapped directions is the arm
    bitmask."""
    H, W = skel.shape
    cluster_set = set(zip(ys.tolist(), xs.tolist()))

    departures: list[tuple[int, int]] = []
    for (py, px) in cluster_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = py + dy, px + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if not skel[ny, nx]:
                    continue
                if labels[ny, nx] == cluster_id:
                    continue
                departures.append((ny, nx))

    cy = float(ys.mean())
    cx = float(xs.mean())

    arms = 0
    for (sy, sx) in departures:
        end = _walk_skeleton(
            skel, labels, cluster_id, start=(sy, sx),
            steps=ARM_WALK_STEPS,
        )
        ey, ex = end
        ddy = ey - cy
        ddx = ex - cx
        d = _snap_direction(ddy, ddx)
        if d is not None:
            arms |= _bit(d)
    return arms


def _walk_skeleton(
    skel: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    start: tuple[int, int],
    steps: int,
) -> tuple[int, int]:
    """BFS-like walk from `start` along the skeleton, never re-entering
    the source cluster. Returns the farthest reachable pixel within
    `steps` graph distance — that direction from the cluster centroid
    indicates the arm orientation."""
    H, W = skel.shape
    visited = {start}
    frontier = deque([(start, 0)])
    farthest = start
    while frontier:
        (cy, cx), depth = frontier.popleft()
        if depth >= steps:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if (ny, nx) in visited:
                    continue
                if not skel[ny, nx]:
                    continue
                if labels[ny, nx] == cluster_id:
                    continue
                visited.add((ny, nx))
                frontier.append(((ny, nx), depth + 1))
                farthest = (ny, nx)
    return farthest


def _snap_direction(dy: float, dx: float) -> Direction | None:
    ay, ax = abs(dy), abs(dx)
    if max(ay, ax) < 1e-6:
        return None
    if ax >= ARM_AXIS_RATIO * ay:
        return "E" if dx > 0 else "W"
    if ay >= ARM_AXIS_RATIO * ax:
        return "S" if dy > 0 else "N"
    return None


def _missing(arms: int) -> list[Direction]:
    out: list[Direction] = []
    if not (arms & ARM_N): out.append("N")
    if not (arms & ARM_E): out.append("E")
    if not (arms & ARM_S): out.append("S")
    if not (arms & ARM_W): out.append("W")
    return out


def _bit(d: Direction) -> int:
    return {"N": ARM_N, "E": ARM_E, "S": ARM_S, "W": ARM_W}[d]


def _classify(arms: int) -> JunctionKind:
    n = bin(arms).count("1")
    if n == 4:
        return "+"
    if n == 3:
        return "T"
    if n == 2:
        if arms in (ARM_N | ARM_E, ARM_E | ARM_S, ARM_S | ARM_W, ARM_W | ARM_N):
            return "L"
        return "I"
    return "?"
