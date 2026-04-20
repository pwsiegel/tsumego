"""End-to-end inference: tight-bbox crop → edge bits + snap params + stones
→ (col, row) for every stone → SGF.

This is the production path. Each step uses a specialized model:
  - YOLO: bbox (single class, just "board")
  - edge classifier: 4 bits (is each side a real board boundary)
  - snap regressor: pitch/origin fractions in the crop
  - stone CNN: stones with pixel positions + colors
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOARD_SIZE = 19


@dataclass(frozen=True)
class PipelineResult:
    stones: list[dict]                  # [{col, row, color, x_px, y_px}, ...]
    sgf: str
    edges: dict[str, bool]
    window: dict[str, int]              # col_min, col_max, row_min, row_max
    pitch: dict[str, float]             # pitch_x_px, pitch_y_px in the original crop
    origin: dict[str, float]            # origin_x_px, origin_y_px
    raw_stones: list[dict]              # original CNN detections, pre-snap
    visible_cols: int
    visible_rows: int


def run_pipeline(crop_bgr: np.ndarray, peak_thresh: float = 0.3) -> PipelineResult:
    from .edge_inference import detect_edges
    from .snap_inference import predict_snap
    from .stone_inference import detect_stones_cnn

    h, w = crop_bgr.shape[:2]
    edges = detect_edges(crop_bgr)
    snap = predict_snap(crop_bgr)

    pitch_x_px = max(1.0, snap["pitch_x"] * w)
    pitch_y_px = max(1.0, snap["pitch_y"] * h)
    origin_x_px = snap["origin_x"] * w
    origin_y_px = snap["origin_y"] * h

    # Visible cells in the crop = how many integer col-steps fit between
    # origin and the far side.
    visible_cols = max(
        2, int(round((w - origin_x_px) / pitch_x_px)) + 1,
    )
    visible_rows = max(
        2, int(round((h - origin_y_px) / pitch_y_px)) + 1,
    )
    visible_cols = min(BOARD_SIZE, visible_cols)
    visible_rows = min(BOARD_SIZE, visible_rows)

    # Determine absolute col_min/row_min from edge bits.
    if edges["left"]:
        col_min = 0
    elif edges["right"]:
        col_min = BOARD_SIZE - visible_cols
    else:
        col_min = 0  # default when neither edge is a boundary
    if edges["top"]:
        row_min = 0
    elif edges["bottom"]:
        row_min = BOARD_SIZE - visible_rows
    else:
        row_min = 0

    col_max = min(BOARD_SIZE - 1, col_min + visible_cols - 1)
    row_max = min(BOARD_SIZE - 1, row_min + visible_rows - 1)

    raw = detect_stones_cnn(crop_bgr, peak_thresh=peak_thresh)
    stones: list[dict] = []
    for s in raw:
        c_local = int(round((s["x"] - origin_x_px) / pitch_x_px))
        r_local = int(round((s["y"] - origin_y_px) / pitch_y_px))
        col = max(col_min, min(col_max, col_min + c_local))
        row = max(row_min, min(row_max, row_min + r_local))
        stones.append({
            "col": col, "row": row, "color": s["color"],
            "x_px": s["x"], "y_px": s["y"],
        })

    sgf = _format_sgf(stones)

    return PipelineResult(
        stones=stones,
        sgf=sgf,
        edges=edges,
        window={"col_min": col_min, "col_max": col_max,
                "row_min": row_min, "row_max": row_max},
        pitch={"x_px": pitch_x_px, "y_px": pitch_y_px},
        origin={"x_px": origin_x_px, "y_px": origin_y_px},
        raw_stones=raw,
        visible_cols=visible_cols,
        visible_rows=visible_rows,
    )


def _format_sgf(stones: list[dict]) -> str:
    def coord(col: int, row: int) -> str:
        return f"{chr(ord('a') + col)}{chr(ord('a') + row)}"
    black = [(s["col"], s["row"]) for s in stones if s["color"] == "B"]
    white = [(s["col"], s["row"]) for s in stones if s["color"] == "W"]
    parts = [f"(;GM[1]FF[4]SZ[{BOARD_SIZE}]"]
    if black:
        parts.append("AB" + "".join(f"[{coord(c, r)}]" for c, r in black))
    if white:
        parts.append("AW" + "".join(f"[{coord(c, r)}]" for c, r in white))
    parts.append(")")
    return "".join(parts)
