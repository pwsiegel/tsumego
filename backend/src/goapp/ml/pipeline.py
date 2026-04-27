"""Shared orchestration: geometry resolution, discretization, board cropping.

These functions tie together the individual ML modules (edge detection,
pitch measurement, stone detection, discretization) into the full
detect-and-discretize pipeline used by both the API routes and CLI tools.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from .board_detect.detect import BoardBBox, ModelNotLoaded, detect_boards_yolo

from ..paths import BBOX_TEST_DIR


BOARD_CROP_PAD = 10  # pixels of safety context around each YOLO bbox

# A real stone fills most of its cell; a hoshi (printed star point) is a
# small dot. Anything detected with bbox radius below this fraction of the
# pitch is assumed to be a hoshi false positive and dropped.
HOSHI_RADIUS_FRAC = 0.28


def _page_bboxes(page_idx: int):
    """Load the page, run YOLO, return (img, bboxes)."""
    import cv2
    import numpy as np
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="could not decode page image")
    try:
        bboxes = _page_bboxes_cached(str(path), path.stat().st_mtime_ns)
    except ModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return img, bboxes


@lru_cache(maxsize=256)
def _page_bboxes_cached(path_str: str, _mtime_ns: int):
    import cv2
    import numpy as np
    arr = np.frombuffer(Path(path_str).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return detect_boards_yolo(img)


def _board_crop(page_idx: int, bbox_idx: int):
    """Return (crop_bgr, (x0, y0, x1, y1)_in_page, bbox) for one detected board.

    Crop is the YOLO bbox plus BOARD_CROP_PAD on each side. Use
    `_page_bboxes(page_idx)` to also get the full page image (needed by
    edge detection to peek outside the crop)."""
    img, bboxes = _page_bboxes(page_idx)
    if bbox_idx < 0 or bbox_idx >= len(bboxes):
        raise HTTPException(
            status_code=404,
            detail=f"bbox {bbox_idx} not found on page {page_idx}",
        )
    b = bboxes[bbox_idx]
    h, w = img.shape[:2]
    P = BOARD_CROP_PAD
    x0 = max(0, b.x0 - P); y0 = max(0, b.y0 - P)
    x1 = min(w, b.x1 + P); y1 = min(h, b.y1 + P)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1), b


def discretize_crop(crop_bgr, peak_thresh: float = 0.3):
    """Run the full detect→discretize pipeline on a raw crop.

    Returns the raw DiscretizationResult plus the edges dict — callers
    that need a Pydantic response shape (the API) wrap further; the
    validation runner consumes this directly. Stone-detector availability
    is the caller's responsibility (we'll let StoneModelNotLoaded
    propagate).
    """
    from .discretize.discretize import discretize
    from .edge_detect.skeleton import decide_edges
    from .edge_detect.tjunction import main_grid_bbox
    from .segments.detect import detect_segments
    from .segments.reason import fit_lattice_fused
    from .stone_detect.clean import filter_to_grid_bbox, paint_out_stones
    from .stone_detect.detect import detect_stones_cnn

    h, w = crop_bgr.shape[:2]
    stones = detect_stones_cnn(crop_bgr, peak_thresh=peak_thresh)

    # Caption / page-number false positives drop here so they don't
    # pollute the lattice fit, segment paint-out, edge classifier,
    # or the discretized output.
    gbb = main_grid_bbox(crop_bgr)
    stones = filter_to_grid_bbox(stones, gbb)

    # Skeleton-based intersections (T/L/+ junctions of the painted-out
    # binary skeleton, with paint-disc artifacts already filtered) are
    # used for both edge classification and as the intersection signal
    # for the lattice fit. The CNN ix detector was trained on
    # stone-free synth and fires on stone-rim arcs at ~1 stone-radius
    # off real grid rows, which poisons the pitch/origin estimate.
    try:
        skel = decide_edges(crop_bgr, stones)
        edges = skel.edges
        ix_centers = [(j.x, j.y) for j in skel.junctions]
    except Exception:
        edges = {"left": False, "right": False, "top": False, "bottom": False}
        ix_centers = []

    # Paint stones out before segment extraction — stone outlines otherwise
    # produce dense short segments that pollute the per-axis position pool.
    cleaned_crop = paint_out_stones(crop_bgr, stones)
    segs = detect_segments(cleaned_crop)

    # Restrict lattice-fit signals to the main grid bbox. Page-binding
    # artifacts and caption rules sit outside the grid CC and would
    # otherwise anchor origin one pitch off — see lab notebook for
    # the tesuji-book right-shift case.
    if gbb is not None:
        gx0, gy0, gx1, gy1 = gbb
        def _seg_in_gbb(s) -> bool:
            mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
            return gx0 <= mx <= gx1 and gy0 <= my <= gy1
        segs_for_fit = [s for s in segs if _seg_in_gbb(s)]
        ix_for_fit = [(x, y) for x, y in ix_centers
                      if gx0 <= x <= gx1 and gy0 <= y <= gy1]
    else:
        segs_for_fit = segs
        ix_for_fit = ix_centers

    fused = fit_lattice_fused(
        segs_for_fit,
        stone_centers=[(s["x"], s["y"]) for s in stones],
        intersection_centers=ix_for_fit,
        crop_w=w, crop_h=h,
        stone_radii=[s["r"] for s in stones if s.get("r")],
    )
    pitch_x, pitch_y = fused.x.pitch, fused.y.pitch
    ox, oy = fused.x.origin, fused.y.origin
    if pitch_x is not None and pitch_y is not None and pitch_x > 0 and pitch_y > 0:
        BOARD_MAX = 18
        top_b = (oy - pitch_y * 0.5) if oy is not None else -1e9
        bot_b = min(h, oy + BOARD_MAX * pitch_y + pitch_y * 0.5) if oy is not None else 1e9
        left_b = (ox - pitch_x * 0.5) if ox is not None else -1e9
        right_b = min(w, ox + BOARD_MAX * pitch_x + pitch_x * 0.5) if ox is not None else 1e9
        stones = [
            s for s in stones
            if top_b <= s["y"] <= bot_b and left_b <= s["x"] <= right_b
        ]

        HOSHI = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                 (15, 3), (15, 9), (15, 15)}
        if ox is not None and oy is not None:
            vc = max(1, min(19, int((w - 1 - ox) / pitch_x) + 1))
            vr = max(1, min(19, int((h - 1 - oy) / pitch_y) + 1))
            cmin = 0 if edges.get("left") else (19 - vc if edges.get("right") else max(0, (19 - vc) // 2))
            rmin = 0 if edges.get("top") else (19 - vr if edges.get("bottom") else max(0, (19 - vr) // 2))
            min_pitch = min(pitch_x, pitch_y)
            hoshi_r = HOSHI_RADIUS_FRAC * min_pitch

            def _not_hoshi(s: dict) -> bool:
                cl = max(0, min(vc - 1, int(round((s["x"] - ox) / pitch_x))))
                rl = max(0, min(vr - 1, int(round((s["y"] - oy) / pitch_y))))
                if (cmin + cl, rmin + rl) in HOSHI and s.get("r", 0) < hoshi_r:
                    return False
                return True
            stones = [s for s in stones if _not_hoshi(s)]

    pitch = (pitch_x + pitch_y) / 2 if pitch_x and pitch_y else pitch_x or pitch_y
    d = discretize(
        stones, w, h, edges=edges,
        cell_size_override=pitch,
        pitch_x_override=pitch_x, pitch_y_override=pitch_y,
        origin_x_override=ox, origin_y_override=oy,
    )
    return d, edges


def _discretize_board(page_idx: int, bbox_idx: int, peak_thresh: float = 0.3):
    """Run the full detect→discretize pipeline on one bbox."""
    from .stone_detect.detect import StoneModelNotLoaded
    from .stone_detect.detect import model_available as stone_model_available
    from ..api.pdf.schemas import BoardDiscretizeLocal, DiscretizedStoneOut

    if not stone_model_available():
        raise HTTPException(status_code=503, detail="stone detector model not trained")
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    h, w = crop.shape[:2]
    try:
        d, edges = discretize_crop(crop, peak_thresh=peak_thresh)
    except StoneModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return BoardDiscretizeLocal(
        page_idx=page_idx, bbox_idx=bbox_idx,
        crop_width=w, crop_height=h,
        cell_size=d.cell_size,
        origin_x=d.origin_x, origin_y=d.origin_y,
        visible_cols=d.visible_cols, visible_rows=d.visible_rows,
        col_min=d.col_min, row_min=d.row_min,
        edges=edges,
        stones=[DiscretizedStoneOut(
            x=s.x, y=s.y, color=s.color, conf=s.conf,
            col_local=s.col_local, row_local=s.row_local,
            col=s.col, row=s.row,
        ) for s in d.stones],
    )
