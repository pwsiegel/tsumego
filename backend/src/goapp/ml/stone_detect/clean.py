from __future__ import annotations

import cv2
import numpy as np

_ANNULUS_INNER = 1.45
_ANNULUS_OUTER = 2.2
# The stone detector returns a "core" radius from the heatmap peak, which is
# noticeably smaller than the stone's visible outer outline. Paint at a
# multiple of the detected radius so the outline ring gets fully covered;
# otherwise it survives and creates sub-pitch peaks that confuse the line
# detector. The additive floor protects very small detections.
_FILL_RADIUS_SCALE = 1.25
_FILL_RADIUS_FLOOR = 4


def filter_to_grid_bbox(
    stones: list[dict],
    grid_bbox: tuple[int, int, int, int] | None,
    margin: float | None = None,
) -> list[dict]:
    """Drop YOLO false positives that sit clearly outside the main
    grid CC bbox — typically caption glyphs ("o" in "problem 14"),
    page numbers, or border text. `margin` defaults to the median
    detected stone radius so a real stone whose center sits flush
    with the outer frame survives."""
    if grid_bbox is None or not stones:
        return stones
    if margin is None:
        import statistics
        margin = float(statistics.median(s["r"] for s in stones))
    x0, y0, x1, y1 = grid_bbox
    return [
        s for s in stones
        if (x0 - margin) <= s["x"] <= (x1 + margin)
        and (y0 - margin) <= s["y"] <= (y1 + margin)
    ]


def paint_radius(r: float) -> int:
    """Pixel radius of the disc that paint_out_stones writes for a
    stone of detector radius `r`. Exposed so other code can mask
    painted regions (e.g. drop skeleton junctions whose centroid sits
    inside a painted disc — those are paint-boundary artifacts, not
    real grid junctions)."""
    return max(int(round(r)) + _FILL_RADIUS_FLOOR, int(round(r * _FILL_RADIUS_SCALE)))


def paint_out_stones(
    crop: np.ndarray, stones: list[dict]
) -> np.ndarray:
    """Replace each detected stone with the local board color.

    For every stone we sample a thin annulus just outside the stone radius,
    take the median color of those pixels, and paint a filled disc at the
    stone location. This removes stone edges that would otherwise confuse
    classical line / grid detectors.
    """
    out = crop.copy()
    h, w = crop.shape[:2]
    for s in stones:
        cx = int(round(s["x"]))
        cy = int(round(s["y"]))
        r = int(round(s["r"]))
        if r <= 0:
            continue
        paint_r = paint_radius(r)
        # Sample bg color from outside the paint zone so the median doesn't
        # include outline-ring pixels we're about to overwrite.
        ri = max(paint_r + 2, int(round(r * _ANNULUS_INNER)))
        ro = max(ri + 1, int(round(r * _ANNULUS_OUTER)))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), ro, 255, -1)
        cv2.circle(mask, (cx, cy), ri, 0, -1)
        sample = out[mask > 0]
        if sample.size == 0:
            continue
        bg = np.median(sample, axis=0)
        if out.ndim == 2:
            color: float | tuple[float, ...] = float(bg)
        else:
            color = tuple(float(c) for c in bg)
        cv2.circle(out, (cx, cy), paint_r, color, -1)
    return out
