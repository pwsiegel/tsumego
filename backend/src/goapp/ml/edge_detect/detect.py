"""Two-signal edge detection on board crops.

Decides which sides of a bbox crop are real board boundaries vs. interior
grid rows/columns where the bbox cuts across mid-board.

We combine two complementary signals on the fitted lattice:

  * **Corner topology (window-cut signal).** At a real edge the
    perpendicular lines terminate at the outermost line (L-junction);
    at a window cut they continue past (T-junction). Counting
    perpendicular Hough segments that extend past the outermost
    detected line tells us "window cut" when it fires.

  * **Thickness / darkness (real-edge signal).** Real outer board
    frames are typically drawn thicker (hm2-style heavy frames) or
    darker (cho-chikun-style heavy frames) than interior grid lines.
    When the outermost lattice line is meaningfully thicker OR darker
    than the next inner line, that's positive evidence for real edge.

Combine: a side is a real edge iff thickness votes yes OR topology
doesn't fire window-cut. Topology alone fails on partial diagrams where
the space past the outer line is just whitespace until a far-away
caption — it has nothing to register against. Thickness alone fails on
thin-frame books like tesuji where the outer line is drawn the same
weight as interior lines. Stacked, they cover both.

Followed by a 19-line sanity rule: a real Go board has exactly 19 lines
per axis, so when the lattice fits fewer than 19, at most one of the
two opposite sides can be real.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..segments.detect import Segment


# A segment counts as horizontal if |dy| < this * |dx| (and vice-versa).
# Hough segments on grid lines are very close to axis-aligned; this
# tolerates slight scan skew without admitting diagonal noise.
AXIS_ALIGN_TOL = 0.15

# How far past the outermost detected line a perpendicular segment must
# extend (as fraction of pitch) before it counts as "continuing past."
# Smaller values catch ticks that are basically at the line; we want
# segments that genuinely cross outward.
PAST_MARGIN_FRAC = 0.4

# When deduplicating extending segments by row/column position, segments
# whose centers are within this fraction of pitch are treated as the
# same lattice line.
DEDUPE_FRAC = 0.5

# Number of distinct perpendicular lines that must extend past the
# outermost detected line before we call it a window cut. A real edge
# can have 0 or 1 spurious extenders (text strokes, stone artifacts);
# 2+ distinct rows or columns extending past is the window-cut signature.
WINDOW_CUT_THRESHOLD = 2

# Adaptive-threshold params for the thickness probe — same values used
# by the segment detector so faint scanned grid lines survive without
# the outer frame melting into background.
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8

# Strip width perpendicular to a line, as a fraction of pitch. Wide
# enough to capture the line + its halo but narrow enough that two
# adjacent lattice lines don't both fall inside.
THICKNESS_STRIP_FRAC = 0.4

# Length along a line direction to sample, as a fraction of the lattice
# extent on the perpendicular axis. Central fraction so corners (where
# two lines meet and density doubles) don't dominate the measurement.
THICKNESS_LENGTH_FRAC = 0.6

# Density threshold (above the strip's max) for "this column/row is part
# of the line." Used to measure line thickness as a contiguous run.
THICKNESS_DENSITY_FRAC = 0.5
THICKNESS_DENSITY_FLOOR = 0.25

# A line is "thicker" than the reference if its run is at least this
# many times longer. 1.5× picks up hm2 frames (drawn ~2× interior) but
# rejects thin-frame books like tesuji where outer ≈ interior.
THICKNESS_RATIO_GATE = 1.5

# A line is "darker" than the reference if its median grayscale is at
# least this many gray levels lower. Cho-chikun frames are drawn 30+
# levels darker; 15 leaves a comfortable margin against scan noise.
DARKNESS_DELTA_GATE = 15.0


class EdgeModelNotLoaded(RuntimeError):
    """Retained for API compatibility. Topology detector never raises this."""


def model_available() -> bool:
    return True


def detect_edges(
    crop_bgr: np.ndarray,
    *,
    pitch_x: float | None = None,
    pitch_y: float | None = None,
    origin_x: float | None = None,
    origin_y: float | None = None,
    segments: list[Segment] | None = None,
) -> dict[str, bool]:
    """Classify each side of the crop as a real board edge or window cut.

    Requires a fitted lattice and the segment list from
    `segments.detect.detect_segments` (run on the stones-painted-out crop
    upstream)."""
    if (
        pitch_x is None or pitch_y is None
        or origin_x is None or origin_y is None
        or segments is None
    ):
        return {"left": False, "right": False, "top": False, "bottom": False}

    crop_h, crop_w = crop_bgr.shape[:2]
    n_cols = max(0, int(np.floor((crop_w - 1 - origin_x) / pitch_x)))
    n_rows = max(0, int(np.floor((crop_h - 1 - origin_y) / pitch_y)))
    outer_left = origin_x
    outer_right = origin_x + n_cols * pitch_x
    outer_top = origin_y
    outer_bottom = origin_y + n_rows * pitch_y

    h_segs = [s for s in segments if _is_horizontal(s)]
    v_segs = [s for s in segments if _is_vertical(s)]

    # Tolerance bands for "this segment lies within the lattice's
    # perpendicular extent" — half a pitch of slack on each end picks up
    # corner ticks without admitting captions far above/below.
    y_band = (outer_top - 0.5 * pitch_y, outer_bottom + 0.5 * pitch_y)
    x_band = (outer_left - 0.5 * pitch_x, outer_right + 0.5 * pitch_x)

    margin_x = PAST_MARGIN_FRAC * pitch_x
    margin_y = PAST_MARGIN_FRAC * pitch_y

    rows_past_left = _distinct_rows_past(
        h_segs, axis_min=outer_left - margin_x, direction="left",
        perp_band=y_band, dedupe=DEDUPE_FRAC * pitch_y,
    )
    rows_past_right = _distinct_rows_past(
        h_segs, axis_min=outer_right + margin_x, direction="right",
        perp_band=y_band, dedupe=DEDUPE_FRAC * pitch_y,
    )
    cols_past_top = _distinct_cols_past(
        v_segs, axis_min=outer_top - margin_y, direction="top",
        perp_band=x_band, dedupe=DEDUPE_FRAC * pitch_x,
    )
    cols_past_bottom = _distinct_cols_past(
        v_segs, axis_min=outer_bottom + margin_y, direction="bottom",
        perp_band=x_band, dedupe=DEDUPE_FRAC * pitch_x,
    )

    topology = {
        "left":   rows_past_left   < WINDOW_CUT_THRESHOLD,
        "right":  rows_past_right  < WINDOW_CUT_THRESHOLD,
        "top":    cols_past_top    < WINDOW_CUT_THRESHOLD,
        "bottom": cols_past_bottom < WINDOW_CUT_THRESHOLD,
    }

    # Thickness/darkness vote. If the outermost lattice line is
    # meaningfully thicker OR darker than the first interior line on
    # that side, vote real edge — overrides a topology window-cut call
    # in cases where stray segments (stone outlines, text near the
    # frame) falsely register as "extending past."
    thickness = _thickness_votes(
        crop_bgr,
        outer_left=outer_left, outer_right=outer_right,
        outer_top=outer_top, outer_bottom=outer_bottom,
        pitch_x=pitch_x, pitch_y=pitch_y,
        n_cols=n_cols, n_rows=n_rows,
    )

    edges = {
        side: topology[side] or thickness[side]
        for side in ("left", "right", "top", "bottom")
    }

    # 19-line sanity. A real Go board has exactly 19 lines per axis. If
    # the lattice fits fewer than 19, at most one of the two opposite
    # sides can be a real edge. Topology can fail to flag a window cut
    # when there's just whitespace past the outermost line (caption sits
    # too far below to register as extending segments). Resolve by margin
    # to the crop boundary — real edges sit tight to the YOLO bbox; the
    # whitespace before a window cut leaves visibly more room.
    n_lines_x = n_cols + 1
    n_lines_y = n_rows + 1
    if n_lines_x < 19 and edges["left"] and edges["right"]:
        left_margin = outer_left
        right_margin = (crop_w - 1) - outer_right
        if left_margin <= right_margin:
            edges["right"] = False
        else:
            edges["left"] = False
    if n_lines_y < 19 and edges["top"] and edges["bottom"]:
        top_margin = outer_top
        bottom_margin = (crop_h - 1) - outer_bottom
        if top_margin <= bottom_margin:
            edges["bottom"] = False
        else:
            edges["top"] = False

    return edges


def _is_horizontal(s: Segment) -> bool:
    dx = abs(s.x2 - s.x1)
    dy = abs(s.y2 - s.y1)
    return dy <= AXIS_ALIGN_TOL * max(dx, 1e-6)


def _is_vertical(s: Segment) -> bool:
    dx = abs(s.x2 - s.x1)
    dy = abs(s.y2 - s.y1)
    return dx <= AXIS_ALIGN_TOL * max(dy, 1e-6)


def _distinct_rows_past(
    h_segs: list[Segment], *,
    axis_min: float, direction: str,
    perp_band: tuple[float, float], dedupe: float,
) -> int:
    """For horizontal segments crossing past `axis_min` in `direction`
    ('left' = x < axis_min, 'right' = x > axis_min), count distinct row
    positions (segments whose y-centers are within `dedupe` collapse to
    one row)."""
    rows: list[float] = []
    for s in h_segs:
        y_avg = 0.5 * (s.y1 + s.y2)
        if not (perp_band[0] <= y_avg <= perp_band[1]):
            continue
        x_min = min(s.x1, s.x2)
        x_max = max(s.x1, s.x2)
        if direction == "left":
            extends = x_min < axis_min
        else:
            extends = x_max > axis_min
        if extends:
            rows.append(y_avg)
    return _count_distinct(rows, dedupe)


def _distinct_cols_past(
    v_segs: list[Segment], *,
    axis_min: float, direction: str,
    perp_band: tuple[float, float], dedupe: float,
) -> int:
    cols: list[float] = []
    for s in v_segs:
        x_avg = 0.5 * (s.x1 + s.x2)
        if not (perp_band[0] <= x_avg <= perp_band[1]):
            continue
        y_min = min(s.y1, s.y2)
        y_max = max(s.y1, s.y2)
        if direction == "top":
            extends = y_min < axis_min
        else:
            extends = y_max > axis_min
        if extends:
            cols.append(x_avg)
    return _count_distinct(cols, dedupe)


def _count_distinct(positions: list[float], min_gap: float) -> int:
    if not positions:
        return 0
    positions = sorted(positions)
    distinct = 1
    last = positions[0]
    for p in positions[1:]:
        if p - last > min_gap:
            distinct += 1
            last = p
    return distinct


def _thickness_votes(
    crop_bgr: np.ndarray, *,
    outer_left: float, outer_right: float,
    outer_top: float, outer_bottom: float,
    pitch_x: float, pitch_y: float,
    n_cols: int, n_rows: int,
) -> dict[str, bool]:
    """For each side, vote True if the outermost lattice line is
    meaningfully thicker OR darker than the first interior line on
    that axis. Needs at least one interior line as reference; if the
    lattice has only one line on an axis (n=0), no vote on that axis."""
    out = {"left": False, "right": False, "top": False, "bottom": False}
    if n_cols < 1 and n_rows < 1:
        return out

    gray = (
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3
        else crop_bgr
    )
    bi = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C,
    )
    h, w = gray.shape

    # Length range to sample along each line — central fraction of the
    # lattice extent on the perpendicular axis.
    y_mid = 0.5 * (outer_top + outer_bottom)
    x_mid = 0.5 * (outer_left + outer_right)
    y_half = 0.5 * THICKNESS_LENGTH_FRAC * max(outer_bottom - outer_top, 1.0)
    x_half = 0.5 * THICKNESS_LENGTH_FRAC * max(outer_right - outer_left, 1.0)
    y_lo = max(0, int(round(y_mid - y_half)))
    y_hi = min(h, int(round(y_mid + y_half)) + 1)
    x_lo = max(0, int(round(x_mid - x_half)))
    x_hi = min(w, int(round(x_mid + x_half)) + 1)

    half_x = THICKNESS_STRIP_FRAC * pitch_x
    half_y = THICKNESS_STRIP_FRAC * pitch_y

    if n_cols >= 1:
        out["left"] = _vote_thicker_or_darker(
            gray, bi, axis="vertical",
            outer_pos=outer_left, inner_pos=outer_left + pitch_x,
            perp_lo=y_lo, perp_hi=y_hi, half_strip=half_x, dim_max=w,
        )
        out["right"] = _vote_thicker_or_darker(
            gray, bi, axis="vertical",
            outer_pos=outer_right, inner_pos=outer_right - pitch_x,
            perp_lo=y_lo, perp_hi=y_hi, half_strip=half_x, dim_max=w,
        )
    if n_rows >= 1:
        out["top"] = _vote_thicker_or_darker(
            gray, bi, axis="horizontal",
            outer_pos=outer_top, inner_pos=outer_top + pitch_y,
            perp_lo=x_lo, perp_hi=x_hi, half_strip=half_y, dim_max=h,
        )
        out["bottom"] = _vote_thicker_or_darker(
            gray, bi, axis="horizontal",
            outer_pos=outer_bottom, inner_pos=outer_bottom - pitch_y,
            perp_lo=x_lo, perp_hi=x_hi, half_strip=half_y, dim_max=h,
        )
    return out


def _vote_thicker_or_darker(
    gray: np.ndarray, bi: np.ndarray, *, axis: str,
    outer_pos: float, inner_pos: float,
    perp_lo: int, perp_hi: int, half_strip: float, dim_max: int,
) -> bool:
    outer = _measure_line(gray, bi, outer_pos, perp_lo, perp_hi, half_strip, dim_max, axis)
    inner = _measure_line(gray, bi, inner_pos, perp_lo, perp_hi, half_strip, dim_max, axis)
    if outer is None or inner is None:
        return False
    outer_thick, outer_gray = outer
    inner_thick, inner_gray = inner
    if outer_thick >= THICKNESS_RATIO_GATE * max(inner_thick, 1):
        return True
    if (inner_gray - outer_gray) >= DARKNESS_DELTA_GATE:
        return True
    return False


def _measure_line(
    gray: np.ndarray, bi: np.ndarray,
    line_pos: float, perp_lo: int, perp_hi: int,
    half_strip: float, dim_max: int, axis: str,
) -> tuple[int, float] | None:
    """Measure (thickness_px, median_gray) of the dominant line in a
    strip centered on `line_pos`. Returns None if no peak found."""
    if not (0 <= line_pos < dim_max):
        return None
    lo = max(0, int(round(line_pos - half_strip)))
    hi = min(dim_max, int(round(line_pos + half_strip)) + 1)
    if hi <= lo or perp_hi <= perp_lo:
        return None
    if axis == "vertical":
        bi_strip = bi[perp_lo:perp_hi, lo:hi]
        gray_strip = gray[perp_lo:perp_hi, lo:hi]
        density = bi_strip.mean(axis=0) / 255.0
    else:
        bi_strip = bi[lo:hi, perp_lo:perp_hi]
        gray_strip = gray[lo:hi, perp_lo:perp_hi]
        density = bi_strip.mean(axis=1) / 255.0
    if density.size == 0:
        return None
    thr = max(THICKNESS_DENSITY_FLOOR, float(density.max()) * THICKNESS_DENSITY_FRAC)
    above = density >= thr
    if not above.any():
        return None
    runs = _bool_runs(above)
    s, e = max(runs, key=lambda r: r[1] - r[0])
    thickness = e - s + 1
    if axis == "vertical":
        run_gray = gray_strip[:, s:e + 1]
        run_bi = bi_strip[:, s:e + 1]
    else:
        run_gray = gray_strip[s:e + 1, :]
        run_bi = bi_strip[s:e + 1, :]
    mask = run_bi > 0
    if not mask.any():
        return None
    return thickness, float(np.median(run_gray[mask]))


def _bool_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            s = i
            while i < n and arr[i]:
                i += 1
            runs.append((s, i - 1))
        else:
            i += 1
    return runs
