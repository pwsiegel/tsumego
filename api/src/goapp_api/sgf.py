"""Convert detected stones on a crop into an SGF problem.

Pipeline:
  1. Cluster the detected stones' x-centers and y-centers to estimate grid
     pitch (spacing between columns/rows). Robust to grid-line occlusion by
     annotation glyphs because it only looks at stone positions.
  2. Inspect each of the crop's four edges to decide whether that edge is a
     real board boundary (A, T, 1, or 19) or a mid-board cut. Signal: a
     thick line parallel to the edge near it, with perpendicular grid lines
     stopping there instead of running off-crop.
  3. Anchor the grid on whichever boundary edges we detected (if any), then
     snap each stone to the nearest intersection on a notional 19x19.
  4. Emit SGF with SZ[19], AB[..] for black setup, AW[..] for white.

If no boundary edges are confidently detected, we still return SGF using
relative coordinates centered in the board — the stones keep their relative
positions, just their absolute 19x19 anchoring is a best guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

BOARD_SIZE = 19


@dataclass(frozen=True)
class BoardMapping:
    pitch: float                  # grid spacing in crop pixels
    origin_x: float               # x-pixel corresponding to column 0 (A)
    origin_y: float               # y-pixel corresponding to row 0 (top of board)
    col_min: int                  # leftmost column index visible in crop
    col_max: int                  # rightmost (inclusive)
    row_min: int                  # topmost row index visible in crop
    row_max: int                  # bottom
    edges_detected: dict          # {'left': bool, 'right': bool, 'top': bool, 'bottom': bool}


def stones_to_sgf(
    crop_bgr: np.ndarray,
    stones: list[dict],
) -> tuple[str, BoardMapping]:
    """stones: list of {"x","y","color"}. Returns (sgf_text, mapping)."""
    if not stones:
        return f"(;GM[1]FF[4]SZ[{BOARD_SIZE}])", _trivial_mapping()

    xs = np.array([s["x"] for s in stones], dtype=np.float32)
    ys = np.array([s["y"] for s in stones], dtype=np.float32)
    h, w = crop_bgr.shape[:2]
    stone_pitch = _estimate_pitch(xs, ys, min_dim=min(h, w))

    # The classifier decides IF each side is a real board boundary. The
    # classical line-finder decides WHERE inside the crop that boundary
    # actually sits (YOLO crops include a few pixels of page margin, so the
    # thick border line is not at pixel 0).
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    edges = _classifier_edges(crop_bgr)

    def pos_if(edge_flag: bool, axis: str) -> float | None:
        return _edge_position(gray, stone_pitch, axis) if edge_flag else None

    left_pos = pos_if(edges["left"], "left")
    right_pos = pos_if(edges["right"], "right")
    top_pos = pos_if(edges["top"], "top")
    bottom_pos = pos_if(edges["bottom"], "bottom")

    # When both edges on an axis are detected, derive pitch exactly from
    # (far edge - near edge) / 18 instead of trusting the stone-cluster
    # estimate, which has ~1-2% error and drifts over 18 cells.
    pitch_x = stone_pitch
    if left_pos is not None and right_pos is not None and right_pos > left_pos:
        pitch_x = (right_pos - left_pos) / (BOARD_SIZE - 1)
    pitch_y = stone_pitch
    if top_pos is not None and bottom_pos is not None and bottom_pos > top_pos:
        pitch_y = (bottom_pos - top_pos) / (BOARD_SIZE - 1)
    pitch = (pitch_x + pitch_y) / 2.0  # reported back for debugging

    origin_x, col_min = _anchor_axis(
        stone_positions=xs, pitch=pitch_x, crop_size=w,
        edge_low=edges["left"], edge_high=edges["right"],
        edge_low_pos=left_pos, edge_high_pos=right_pos,
    )
    origin_y, row_min = _anchor_axis(
        stone_positions=ys, pitch=pitch_y, crop_size=h,
        edge_low=edges["top"], edge_high=edges["bottom"],
        edge_low_pos=top_pos, edge_high_pos=bottom_pos,
    )

    col_max = min(BOARD_SIZE - 1, int((w - origin_x) / pitch_x))
    row_max = min(BOARD_SIZE - 1, int((h - origin_y) / pitch_y))

    mapping = BoardMapping(
        pitch=pitch, origin_x=origin_x, origin_y=origin_y,
        col_min=max(0, col_min), col_max=col_max,
        row_min=max(0, row_min), row_max=row_max,
        edges_detected=edges,
    )

    black: list[tuple[int, int]] = []
    white: list[tuple[int, int]] = []
    for s in stones:
        col = int(round((s["x"] - origin_x) / pitch_x))
        row = int(round((s["y"] - origin_y) / pitch_y))
        col = max(0, min(BOARD_SIZE - 1, col))
        row = max(0, min(BOARD_SIZE - 1, row))
        (black if s["color"] == "B" else white).append((col, row))

    sgf = _format_sgf(black, white)
    return sgf, mapping


def _trivial_mapping() -> BoardMapping:
    return BoardMapping(
        pitch=0.0, origin_x=0.0, origin_y=0.0,
        col_min=0, col_max=BOARD_SIZE - 1,
        row_min=0, row_max=BOARD_SIZE - 1,
        edges_detected={"left": False, "right": False, "top": False, "bottom": False},
    )


def _estimate_pitch(xs: np.ndarray, ys: np.ndarray, min_dim: int) -> float:
    """Estimate pixel distance between neighboring grid lines.

    Cluster stone centers into 1D bands (gap threshold scales with crop
    size). Compute diffs between cluster centers, drop any that's below a
    plausible minimum pitch (would require >20 columns across the crop's
    shorter dimension), divide each remaining diff by its likely integer
    multiplier, and take the median.
    """
    gap = max(4.0, float(min_dim) * 0.025)
    # A real pitch can't be much smaller than min_dim / 20; otherwise the
    # board would have >20 cells in that direction, which can't happen on a
    # 19x19. Use this to reject outlier sub-pitch diffs.
    pitch_min = float(min_dim) / 22.0

    def pitch_of(coords: np.ndarray) -> float | None:
        centers = _cluster_1d(coords, gap=gap)
        if len(centers) < 2:
            return None
        diffs = np.diff(centers)
        diffs = diffs[diffs >= pitch_min]
        if diffs.size == 0:
            return None
        smallest = float(np.min(diffs))
        ks = np.maximum(1, np.round(diffs / smallest))
        candidates = diffs / ks
        return float(np.median(candidates))

    px = pitch_of(xs)
    py = pitch_of(ys)
    vals = [p for p in (px, py) if p is not None and p > 0]
    if not vals:
        return max(10.0, float(min_dim) / 18.0)
    return float(np.median(vals))


def _cluster_1d(coords: np.ndarray, gap: float) -> list[float]:
    """Greedy 1D clustering: consecutive sorted values within `gap` of the
    running cluster mean are merged. Returns cluster centers (means)."""
    if coords.size == 0:
        return []
    sorted_c = np.sort(coords)
    clusters: list[list[float]] = [[float(sorted_c[0])]]
    for v in sorted_c[1:]:
        v = float(v)
        if v - np.mean(clusters[-1]) <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


_EDGE_MARGIN_PITCHES = 3.0
_EDGE_RUN_FRACTION = 0.85


def _classifier_edges(crop_bgr: np.ndarray) -> dict[str, bool]:
    """Thin wrapper that defers to the trained edge classifier when
    available, and falls back to "no edges" when the model hasn't been
    trained yet."""
    from .edge_inference import EdgeModelNotLoaded, detect_edges, model_available
    if not model_available():
        return {"left": False, "right": False, "top": False, "bottom": False}
    try:
        return detect_edges(crop_bgr)
    except EdgeModelNotLoaded:
        return {"left": False, "right": False, "top": False, "bottom": False}


def _edge_strip(binary: np.ndarray, axis: str, pitch: float):
    """Return (strip, offset) for the named edge."""
    h, w = binary.shape
    margin = int(max(pitch * _EDGE_MARGIN_PITCHES, 20))
    if axis == "left":
        return binary[:, :min(margin, w)], 0
    if axis == "right":
        off = max(0, w - margin)
        return binary[:, off:], off
    if axis == "top":
        return binary[:min(margin, h), :], 0
    off = max(0, h - margin)
    return binary[off:, :], off


def _longest_run_per_line(strip: np.ndarray, axis: str) -> np.ndarray:
    """For each candidate line position (column if vertical, row if
    horizontal), return the length of its longest unbroken dark run as a
    fraction of the perpendicular dimension."""
    if axis in ("left", "right"):
        # Iterate columns; each column's run length along rows.
        columns = strip.T  # now shape (cols, rows)
        perp_len = strip.shape[0]
    else:
        columns = strip  # each row's run length along cols.
        perp_len = strip.shape[1]

    out = np.zeros(columns.shape[0], dtype=np.float32)
    for i, line in enumerate(columns):
        run = longest = 0
        for v in line:
            if v:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
        out[i] = longest / max(1, perp_len)
    return out


def _detect_board_edges(gray: np.ndarray, pitch: float) -> dict[str, bool]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    out: dict[str, bool] = {}
    for axis in ("left", "right", "top", "bottom"):
        strip, _ = _edge_strip(binary, axis, pitch)
        if strip.size == 0:
            out[axis] = False
            continue
        runs = _longest_run_per_line(strip, axis)
        out[axis] = bool(runs.max() >= _EDGE_RUN_FRACTION)
    return out


def _edge_position(gray: np.ndarray, pitch: float, axis: str) -> float | None:
    """Return the crop-pixel coordinate of the board boundary on the named
    edge. For left/top edges this is the FIRST line (closest to the crop
    edge) that's long enough to be the border; for right/bottom edges the
    LAST one. "Longest run" alone is unreliable: interior grid lines span
    the full crop height too and can outrun the actual border by a few
    pixels, causing the anchoring to latch onto col 1 or 2 instead of col
    0."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    strip, offset = _edge_strip(binary, axis, pitch)
    if strip.size == 0:
        return None
    runs = _longest_run_per_line(strip, axis)
    hits = np.where(runs >= _EDGE_RUN_FRACTION)[0]
    if hits.size == 0:
        return None
    idx = int(hits[0]) if axis in ("left", "top") else int(hits[-1])
    return float(offset + idx)


def _anchor_axis(
    stone_positions: np.ndarray,
    pitch: float,
    crop_size: int,
    edge_low: bool,
    edge_high: bool,
    edge_low_pos: float | None,
    edge_high_pos: float | None,
) -> tuple[float, int]:
    """Return (origin_pixel, first_visible_index) for one axis.

    `origin_pixel` is the crop-pixel coordinate of board-index 0 (A or top).
    """
    if edge_low and edge_low_pos is not None:
        return float(edge_low_pos), 0
    if edge_high and edge_high_pos is not None:
        # The detected line is column 18 (or row 18). Work backwards.
        origin = edge_high_pos - (BOARD_SIZE - 1) * pitch
        first_visible = int(max(0, np.floor(-origin / pitch))) if origin < 0 else 0
        return float(origin), first_visible

    # No edge on this axis: default is "the leftmost/topmost stone cluster is
    # at index 0 of the visible region". This is right for typical
    # corner/side problems shown in Go books. Middle-of-board problems will
    # end up off by a few columns, which the user can correct in the viewer.
    if stone_positions.size == 0:
        return 0.0, 0
    centers = _cluster_1d(stone_positions, gap=pitch * 0.5)
    leftmost = centers[0] if centers else float(np.min(stone_positions))
    return float(leftmost), 0


def _format_sgf(black: list[tuple[int, int]], white: list[tuple[int, int]]) -> str:
    def coord(col: int, row: int) -> str:
        # SGF uses a=0 ... s=18 for board size 19. lowercase.
        return f"{chr(ord('a') + col)}{chr(ord('a') + row)}"

    parts = [f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]"]
    if black:
        parts.append("AB" + "".join(f"[{coord(c, r)}]" for c, r in black))
    if white:
        parts.append("AW" + "".join(f"[{coord(c, r)}]" for c, r in white))
    parts.append(")")
    return "".join(parts)
