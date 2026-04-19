"""Board detection: find Go boards in a rendered PDF page image."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardBBox:
    x0: int
    y0: int
    x1: int
    y1: int
    h_lines: int  # horizontal-line pixel density × 1000 inside the bbox
    v_lines: int  # vertical-line pixel density × 1000 inside the bbox


def detect_boards(image_bgr: np.ndarray) -> list[BoardBBox]:
    """Find rectangular Go-board regions in a page image.

    Line-based: a Go grid is the ONLY thing that combines a set of parallel
    horizontal lines AND a set of parallel vertical lines in the same region.
    Dilate each line mask along the axis perpendicular to its lines so each
    board's stack of parallel lines fuses into a solid stripe. AND the two
    dilated masks — the intersection is exactly the rectangular region of
    each grid. Connected components split that by board.

    Robust to: stones covering intersections (line masks survive), partial
    boards with no outer frame (no frame required), tiled layouts (gaps
    between boards lack lines in one or both axes).
    """
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    h, w = gray.shape
    log.info("detect_boards: image %dx%d", w, h)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10,
    )

    hk = max(10, w // 30)
    vk = max(10, h // 30)
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))

    # Estimate cell size from the spacing between consecutive horizontal lines.
    # Use diffs > 3 to skip within-line adjacency noise.
    cell_est = _estimate_cell_size(h_mask, v_mask, w, h)
    log.info("detect_boards: estimated cell size=%d", cell_est)

    # Bridge adjacent parallel lines into solid stripes. Kernel ~1 cell is
    # enough to connect consecutive grid lines; larger than that risks
    # merging separate boards.
    bridge = max(5, int(cell_est * 1.2))
    h_stripe = cv2.dilate(
        h_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)),
    )
    v_stripe = cv2.dilate(
        v_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)),
    )
    grid_region = cv2.bitwise_and(h_stripe, v_stripe)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(grid_region)
    log.info("detect_boards: %d grid regions", num_labels - 1)

    # Solid-content mask used to extend bboxes to adjacent stone clusters.
    # Otsu + morphological opening with a cell-sized ellipse keeps only blobs
    # of roughly stone size (filled circles) and removes thin features
    # (text strokes, grid lines).
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    stone_open_k = max(5, int(cell_est * 0.4))
    stone_mask = cv2.morphologyEx(
        otsu, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stone_open_k, stone_open_k)),
    )

    results: list[BoardBBox] = []
    page_area = w * h
    for i in range(1, num_labels):
        x, y, cw, ch, _area = stats[i]
        bbox_area = cw * ch
        if bbox_area < page_area * 0.005 or bbox_area > page_area * 0.8:
            continue

        # Shrink to the actual extent of grid-line pixels (trim dilation halo).
        region_h = h_mask[y:y + ch, x:x + cw]
        region_v = v_mask[y:y + ch, x:x + cw]
        rows_with_line = np.where(np.any(region_h, axis=1))[0]
        cols_with_line = np.where(np.any(region_v, axis=0))[0]
        if rows_with_line.size < 2 or cols_with_line.size < 2:
            continue

        # Require enough distinct grid lines to rule out frame-only rectangles.
        # Count "groups" of consecutive rows/cols (each grid line is 1-3px thick).
        h_line_groups = _count_line_groups(rows_with_line)
        v_line_groups = _count_line_groups(cols_with_line)
        if h_line_groups < 4 or v_line_groups < 4:
            continue

        y0 = int(y + rows_with_line.min())
        y1 = int(y + rows_with_line.max())
        x0 = int(x + cols_with_line.min())
        x1 = int(x + cols_with_line.max())

        # Extend bbox to include adjacent stone clusters. Look within a
        # ~2-cell margin on each side for solid content; if found, grow the
        # bbox to include it. Handles books where stones are packed densely
        # enough that grid lines in that region are lost to morphological
        # opening, so line detection alone misses the stone-populated edge.
        margin = int(cell_est * 2)
        sx0 = max(0, x0 - margin)
        sy0 = max(0, y0 - margin)
        sx1 = min(w, x1 + margin)
        sy1 = min(h, y1 + margin)
        search = stone_mask[sy0:sy1, sx0:sx1]
        content_rows = np.where(np.any(search, axis=1))[0]
        content_cols = np.where(np.any(search, axis=0))[0]
        if content_rows.size > 0 and content_cols.size > 0:
            cr_min = int(sy0 + content_rows.min())
            cr_max = int(sy0 + content_rows.max())
            cc_min = int(sx0 + content_cols.min())
            cc_max = int(sx0 + content_cols.max())
            y0 = min(y0, cr_min)
            y1 = max(y1, cr_max)
            x0 = min(x0, cc_min)
            x1 = max(x1, cc_max)

        new_w, new_h = x1 - x0, y1 - y0
        if new_w < 10 or new_h < 10:
            continue
        aspect = new_w / new_h
        if not 0.2 < aspect < 5.0:
            continue

        area = new_w * new_h
        h_density = float(np.count_nonzero(h_mask[y0:y1, x0:x1])) / area
        v_density = float(np.count_nonzero(v_mask[y0:y1, x0:x1])) / area

        results.append(BoardBBox(
            x0=x0, y0=y0, x1=x1, y1=y1,
            h_lines=int(h_density * 1000),
            v_lines=int(v_density * 1000),
        ))

    results = _dedupe_bboxes(results)
    results.sort(key=lambda b: -((b.x1 - b.x0) * (b.y1 - b.y0)))
    log.info("detect_boards: %d boards: %s", len(results), results)
    return results


def _count_line_groups(indices: np.ndarray, max_gap: int = 3) -> int:
    """Count distinct clusters of consecutive indices — each is one grid line
    (grid lines are 1-3px thick so occupy a few adjacent rows/cols)."""
    if indices.size == 0:
        return 0
    groups = 1
    for i in range(1, indices.size):
        if indices[i] - indices[i - 1] > max_gap:
            groups += 1
    return groups


def _estimate_cell_size(
    h_mask: np.ndarray, v_mask: np.ndarray, page_w: int, page_h: int,
) -> int:
    """Median spacing between consecutive rows with horizontal-line content,
    fallbacks included. A Go grid has regular row-to-row spacing; that's
    the cell size."""
    rows = np.where(np.any(h_mask, axis=1))[0]
    if rows.size >= 2:
        diffs = np.diff(rows)
        reasonable = diffs[(diffs > 3) & (diffs < page_h // 3)]
        if reasonable.size > 0:
            return int(np.median(reasonable))
    cols = np.where(np.any(v_mask, axis=0))[0]
    if cols.size >= 2:
        diffs = np.diff(cols)
        reasonable = diffs[(diffs > 3) & (diffs < page_w // 3)]
        if reasonable.size > 0:
            return int(np.median(reasonable))
    return max(20, min(page_w, page_h) // 50)


def _containment(inner: BoardBBox, outer: BoardBBox) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix0, iy0 = max(inner.x0, outer.x0), max(inner.y0, outer.y0)
    ix1, iy1 = min(inner.x1, outer.x1), min(inner.y1, outer.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    inner_area = (inner.x1 - inner.x0) * (inner.y1 - inner.y0)
    return inter / inner_area if inner_area else 0.0


def _dedupe_bboxes(boards: list[BoardBBox]) -> list[BoardBBox]:
    """Drop candidates that are mostly contained in a larger one.

    Rationale: the intersection-cluster pass often finds sub-regions of a
    real board as their own "candidates". If B is ≥70% inside A, A is the
    actual board and B is a spurious subset. Sorting by area descending so
    parents are considered before their children.
    """
    sorted_boards = sorted(
        boards, key=lambda b: -((b.x1 - b.x0) * (b.y1 - b.y0))
    )
    kept: list[BoardBBox] = []
    for b in sorted_boards:
        if any(_containment(b, k) > 0.7 for k in kept):
            continue
        kept.append(b)
    return kept


def detect_boards_debug(image_bgr: np.ndarray) -> bytes:
    """Return a PNG showing the detection pipeline output:
    original image + overlays for line masks and detected bboxes.
    """
    if image_bgr.ndim == 3:
        vis = image_bgr.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )
    hk, vk = max(10, w // 30), max(10, h // 30)
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))

    # Tint the line masks onto the visualization.
    vis[h_mask > 0] = (0, 255, 0)   # green = horizontal
    vis[v_mask > 0] = (0, 0, 255)   # red   = vertical

    # Boxes in blue.
    boards = detect_boards(image_bgr)
    for b in boards:
        cv2.rectangle(vis, (b.x0, b.y0), (b.x1, b.y1), (255, 0, 0), 4)
        label = f"H={b.h_lines} V={b.v_lines}"
        cv2.putText(vis, label, (b.x0 + 8, b.y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise RuntimeError("could not encode debug image")
    return png.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes → BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img
