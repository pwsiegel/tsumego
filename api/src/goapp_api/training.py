"""Training-data management: persist labeled examples to disk.

Paths are sourced from goapp_api.paths (driven by $GOAPP_DATA_DIR, default
~/data/go-app). Nothing lives inside the repo.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

from .paths import BOARDS_DEPRECATED_DIR as BOARDS_DIR
from .paths import DATA_DIR as TRAINING_ROOT
from .paths import STONE_POINTS_DIR, STONE_TASKS_DIR


@dataclass(frozen=True)
class SavedBoardLabel:
    label_id: str
    image_path: Path
    json_path: Path
    bboxes: list[tuple[int, int, int, int]]


def save_board_label(image_bytes: bytes, bboxes: list[tuple[int, int, int, int]]) -> SavedBoardLabel:
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    short = secrets.token_hex(3)
    label_id = f"{stamp}_{short}"
    img_path = BOARDS_DIR / f"{label_id}.png"
    json_path = BOARDS_DIR / f"{label_id}.json"

    img_path.write_bytes(image_bytes)
    json_path.write_text(
        json.dumps({"bboxes": [list(b) for b in bboxes]}, indent=2)
    )
    log.info("saved board label %s with %d bboxes", label_id, len(bboxes))
    return SavedBoardLabel(
        label_id=label_id,
        image_path=img_path,
        json_path=json_path,
        bboxes=bboxes,
    )


def count_board_labels() -> int:
    if not BOARDS_DIR.exists():
        return 0
    return sum(1 for _ in BOARDS_DIR.glob("*.json"))


# -------- stone-point labeling ----------------------------------------------


def _task_id(board_id: str, bbox_idx: int) -> str:
    return f"{board_id}_b{bbox_idx}"


def list_stone_tasks() -> list[dict]:
    """Stone-tuning tasks — only the crops from the most recently ingested
    PDF. The board-detector's training data (BOARDS_DIR) is not surfaced here.
    """
    tasks: list[dict] = []
    if STONE_TASKS_DIR.exists():
        for png_path in sorted(STONE_TASKS_DIR.glob("*.png")):
            tid = png_path.stem
            tasks.append({
                "task_id": tid,
                "source": "auto_detected",
                "labeled": (STONE_POINTS_DIR / f"{tid}.json").exists(),
            })
    return tasks


def get_task_crop(task_id: str) -> bytes:
    """Return PNG bytes for any task_id (boards-sourced or auto-detected)."""
    if task_id.startswith("st_"):
        png_path = STONE_TASKS_DIR / f"{task_id}.png"
        if not png_path.exists():
            raise FileNotFoundError(task_id)
        return png_path.read_bytes()
    import re
    m = re.match(r"^(.+)_b(\d+)$", task_id)
    if not m:
        raise ValueError(f"invalid task_id: {task_id}")
    return get_board_crop(m.group(1), int(m.group(2)))


def _load_task_crop_array(task_id: str) -> np.ndarray:
    """BGR numpy array for any task_id."""
    if task_id.startswith("st_"):
        png_path = STONE_TASKS_DIR / f"{task_id}.png"
        if not png_path.exists():
            raise FileNotFoundError(task_id)
        arr = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode {png_path}")
        return img
    import re
    m = re.match(r"^(.+)_b(\d+)$", task_id)
    if not m:
        raise ValueError(f"invalid task_id: {task_id}")
    return _load_board_crop(m.group(1), int(m.group(2)))


def ingest_pdf_for_stone_tasks_stream(
    pdf_bytes: bytes, source_name: str = "pdf",
):
    """Generator version: yields progress dicts as pages are processed.

    Events:
      {"total_pages": N}              — first event
      {"page": p, "tasks_added": t}   — per-page update
      {"done": True, "total_tasks": t} — final event
    """
    import io
    import shutil
    import pypdfium2 as pdfium
    from .inference import detect_boards_with_edges, model_available

    if not model_available():
        raise RuntimeError(
            "no trained board model available — train first before ingesting"
        )

    if STONE_TASKS_DIR.exists():
        shutil.rmtree(STONE_TASKS_DIR)
    STONE_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    total_pages = len(pdf)
    yield {"total_pages": total_pages, "source": source_name}

    count = 0
    for page_num in range(total_pages):
        page = pdf[page_num]
        pil_img = page.render(scale=2.0).to_pil()
        img_bgr = np.array(pil_img)[..., ::-1].copy()
        try:
            detections = detect_boards_with_edges(img_bgr)
        except Exception as e:
            log.warning("detection failed on page %d: %s", page_num + 1, e)
            detections = []
        for b, edges in detections:
            x0 = max(0, b.x0); y0 = max(0, b.y0)
            x1 = min(img_bgr.shape[1], b.x1); y1 = min(img_bgr.shape[0], b.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img_bgr[y0:y1, x0:x1]
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            short = secrets.token_hex(4)
            task_id = f"st_{stamp}_{short}"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            (STONE_TASKS_DIR / f"{task_id}.png").write_bytes(buf.tobytes())
            (STONE_TASKS_DIR / f"{task_id}.json").write_text(json.dumps({
                "source": source_name, "page": page_num + 1,
                "bbox_in_page": [int(x0), int(y0), int(x1), int(y1)],
                "confidence": b.h_lines / 1000.0,
                "edges": edges,
            }))
            count += 1
        yield {"page": page_num + 1, "tasks_added": count}

    log.info("ingested %d stone tasks from %s", count, source_name)
    yield {"done": True, "total_tasks": count}


def ingest_pdf_for_stone_tasks(pdf_bytes: bytes, source_name: str = "pdf") -> int:
    """Non-streaming wrapper — kept for compatibility with the old endpoint
    signature. Drains the generator and returns the final count."""
    last_count = 0
    for event in ingest_pdf_for_stone_tasks_stream(pdf_bytes, source_name):
        if "total_tasks" in event:
            last_count = event["total_tasks"]
    return last_count


def _load_board_crop(board_id: str, bbox_idx: int) -> np.ndarray:
    """Return the cropped board as a BGR numpy array."""
    json_path = BOARDS_DIR / f"{board_id}.json"
    img_path = BOARDS_DIR / f"{board_id}.png"
    if not json_path.exists() or not img_path.exists():
        raise FileNotFoundError(board_id)
    data = json.loads(json_path.read_text())
    bboxes = data.get("bboxes", [])
    if bbox_idx < 0 or bbox_idx >= len(bboxes):
        raise IndexError(f"bbox_idx {bbox_idx} out of range for {board_id}")
    x0, y0, x1, y1 = (int(v) for v in bboxes[bbox_idx])

    arr = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {img_path}")
    return img[max(0, y0):y1, max(0, x0):x1]


def get_board_crop(board_id: str, bbox_idx: int) -> bytes:
    """Return PNG bytes for the bbox-th crop of the given board label."""
    crop = _load_board_crop(board_id, bbox_idx)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError("encoding crop failed")
    return buf.tobytes()


def detect_stone_circles_for_task(
    task_id: str,
    min_r_frac: float = 0.02,
    max_r_frac: float = 0.15,
    hough_param2: int = 40,
    white_ring_thresh: float = 0.1,
) -> list[dict]:
    """Same as detect_stone_circles but dispatched on task_id."""
    crop = _load_task_crop_array(task_id)
    return _detect_stone_circles_on_crop(
        crop, min_r_frac=min_r_frac, max_r_frac=max_r_frac,
        hough_param2=hough_param2, white_ring_thresh=white_ring_thresh,
    )


def detect_board_grid(board_id: str, bbox_idx: int) -> dict:
    """Detect grid intersection positions within a board crop.

    Returns {'rows': [y0, y1, ...], 'cols': [x0, x1, ...]} where each list is
    the (sub-pixel) center of one grid line in crop-local coordinates. The
    front-end snaps clicks to the nearest (col, row) intersection.

    Strategy: adaptive threshold → morphological opening with long horizontal
    (resp. vertical) structuring element → row-sum (col-sum) of the mask →
    peak-find to locate individual line centers.
    """
    crop = _load_board_crop(board_id, bbox_idx)
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    h, w = gray.shape

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10,
    )
    # Short kernel so partial-board grid lines (broken by stones) still survive.
    hk = max(8, w // 12)
    vk = max(8, h // 12)
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))

    row_sums = np.count_nonzero(h_mask, axis=1).astype(np.float32)
    col_sums = np.count_nonzero(v_mask, axis=0).astype(np.float32)

    # Lower threshold to catch weaker (partially-stone-obscured) grid lines.
    rows = _peak_centers(row_sums, min_gap=max(3, h // 60), rel_thresh=0.2)
    cols = _peak_centers(col_sums, min_gap=max(3, w // 60), rel_thresh=0.2)
    return {"rows": rows, "cols": cols}


def detect_stone_circles(board_id: str, bbox_idx: int) -> list[dict]:
    """Dispatch on (board_id, bbox_idx) — kept for backward compat."""
    return _detect_stone_circles_on_crop(_load_board_crop(board_id, bbox_idx))


def _detect_stone_circles_on_crop(
    crop: np.ndarray,
    min_r_frac: float = 0.02,
    max_r_frac: float = 0.15,
    hough_param2: int = 40,
    white_ring_thresh: float = 0.1,
) -> list[dict]:
    """Detect stone circles separately for black (solid dark blobs) and
    white (light interior regions enclosed by a ring). Grid-cell-interior
    false positives are filtered out via corner detection — a polygon
    approximation of a grid cell collapses to 4-7 vertices, whereas a
    smooth stone outline stays with many.

    Upsamples tiny crops first (book-2-style partial boards can come in
    under 300px high), because the morphology thresholds scale with crop
    size and don't behave on very small inputs.
    """
    target_min = 400
    orig_h, orig_w = crop.shape[:2]
    in_min = min(orig_w, orig_h)
    scale = 1.0
    if in_min < target_min:
        scale = target_min / in_min
        crop = cv2.resize(
            crop, (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Pad with paper-white so stones clipped by the crop boundary still have
    # room for HoughCircles to complete their circular gradient vote.
    work_h, work_w = crop.shape[:2]
    pad = max(20, int(min(work_w, work_h) * 0.05))
    crop = cv2.copyMakeBorder(
        crop, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    min_dim = min(w, h)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Strip long horizontal/vertical lines (grid + frame). Thresholds are
    # per-axis so that wide/short partial-board crops still work: horizontal
    # grid lines span most of the width, vertical lines span most of the
    # height. The kernel length must be longer than any stone's diameter so
    # stones aren't mistaken for lines.
    h_k = max(20, int(w * 0.4))
    v_k = max(20, int(h * 0.4))
    h_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1)),
    )
    v_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k)),
    )
    lines = cv2.bitwise_or(h_lines, v_lines)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    no_lines = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

    open_k = max(3, int(min_dim * 0.005))
    if open_k % 2 == 0:
        open_k += 1
    dark = cv2.morphologyEx(
        no_lines, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    )
    light = cv2.morphologyEx(
        cv2.bitwise_not(binary), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    )
    # Fill small dark features inside light regions (triangles, numbered-move
    # glyphs). Without this, a white stone with a triangle inside has its
    # light interior carved into a non-circular shape that the circularity
    # filter rejects.
    glyph_close_k = max(15, int(min_dim * 0.02))
    if glyph_close_k % 2 == 0:
        glyph_close_k += 1
    light = cv2.morphologyEx(
        light, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glyph_close_k, glyph_close_k)),
    )

    floor_min_r = max(8, int(min_dim * min_r_frac))
    ceil_max_r = max(floor_min_r + 5, int(min_dim * max_r_frac))

    def _roundish(mask: np.ndarray, is_interior: bool) -> list[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if is_interior and (x == 0 or y == 0 or x + cw == w or y + ch == h):
                continue
            r = (cw + ch) / 4.0
            if r < floor_min_r or r > ceil_max_r:
                continue
            aspect = cw / max(1, ch)
            if aspect < 0.55 or aspect > 1.8:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < 0.7:
                continue
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if 3 <= len(approx) <= 7:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            out.append({"x": float(cx), "y": float(cy), "r": float(r), "area": float(area)})
        return out

    cc_cands = _roundish(dark, is_interior=False) + _roundish(light, is_interior=True)

    # HoughCircles handles tightly-clustered stones (connected-component
    # detection can't separate touching blobs). Use a radius range based on
    # our cc candidates when available, else fall back to a size guess.
    if cc_cands:
        rs = sorted(c["r"] for c in cc_cands)
        med_r = rs[len(rs) // 2]
        min_r_h = max(floor_min_r, int(med_r * 0.5))
        max_r_h = min(ceil_max_r, max(min_r_h + 5, int(med_r * 1.8)))
    else:
        min_r_h = floor_min_r
        max_r_h = ceil_max_r

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    hough_raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=max(8, int(min_r_h * 1.5)),
        param1=80, param2=hough_param2,
        minRadius=min_r_h, maxRadius=max_r_h,
    )

    # Second pass: fill small light holes (triangle glyphs) AND white-stone
    # interiors so these become solid dark disks, matching HoughCircles'
    # gradient-direction assumption. Close-kernel ≈ stone interior diameter.
    # This catches isolated white stones that regular Hough misses.
    fill_k = max(15, min_r_h * 2 - 5)
    if fill_k % 2 == 0:
        fill_k += 1
    binary_filled = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_k, fill_k)),
    )
    # Convert to grayscale where dark content is 0 and background is 255.
    filled_gray = cv2.bitwise_not(binary_filled)
    filled_blurred = cv2.GaussianBlur(filled_gray, (5, 5), 1.5)
    hough_filled = cv2.HoughCircles(
        filled_blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=max(8, int(min_r_h * 1.5)),
        param1=80, param2=hough_param2,
        minRadius=min_r_h, maxRadius=max_r_h,
    )

    # Concatenate both passes' detections for verification.
    hough_all = []
    if hough_raw is not None:
        hough_all.extend(hough_raw[0].tolist())
    if hough_filled is not None:
        hough_all.extend(hough_filled[0].tolist())
    hough = [hough_all] if hough_all else None
    hough_cands: list[dict] = []
    if hough is not None:
        n_samples = 32
        # For each angle around the circle, sample pixels along a wide
        # radial range (Hough's detected radius is often 15%+ off from the
        # actual outline). Any dark pixel along that radial line counts the
        # angle as "dark".
        radial_frac = np.linspace(0.65, 1.25, 13)
        for c in hough[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            angles_dark = 0
            for i in range(n_samples):
                theta = 2 * np.pi * i / n_samples
                found = False
                for rf in radial_frac:
                    sx = int(cx + r * rf * np.cos(theta))
                    sy = int(cy + r * rf * np.sin(theta))
                    if 0 <= sx < w and 0 <= sy < h and gray[sy, sx] < 140:
                        found = True
                        break
                if found:
                    angles_dark += 1
            outline_frac = angles_dark / n_samples
            if outline_frac >= white_ring_thresh:
                hough_cands.append({"x": cx, "y": cy, "r": r, "area": float(np.pi * r * r)})

    # Merge cc_cands + hough_cands; dedupe circles with overlapping centers.
    merged: list[dict] = []
    for c in cc_cands + hough_cands:
        dup = False
        for m in merged:
            if np.hypot(c["x"] - m["x"], c["y"] - m["y"]) < max(c["r"], m["r"]) * 0.7:
                dup = True
                break
        if not dup:
            merged.append(c)

    if not merged:
        return []
    areas = np.array([c["area"] for c in merged])
    med = float(np.median(areas))
    stones = [c for c in merged if med * 0.25 <= c["area"] <= med * 4.0]
    # Remove padding offset, scale back to original crop pixel space. Drop
    # any detection whose center landed in the pad margin. Classify each
    # circle as B or W by the median luminance of its inner half (robust to
    # glyphs like triangles or move numbers drawn inside the stone).
    out: list[dict] = []
    for s in stones:
        # Color classification in padded coordinates (gray has the pad too).
        r_inner = max(2, int(s["r"] * 0.5))
        cy_p, cx_p = int(s["y"]), int(s["x"])
        y_lo = max(0, cy_p - r_inner); y_hi = min(h, cy_p + r_inner + 1)
        x_lo = max(0, cx_p - r_inner); x_hi = min(w, cx_p + r_inner + 1)
        if y_hi > y_lo and x_hi > x_lo:
            patch = gray[y_lo:y_hi, x_lo:x_hi]
            median_lum = float(np.median(patch))
            color = "B" if median_lum < 128 else "W"
        else:
            color = "W"

        x = (s["x"] - pad) / scale
        y = (s["y"] - pad) / scale
        r = s["r"] / scale
        if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
            continue
        out.append({"x": x, "y": y, "r": r, "color": color})
    return out


def _peak_centers(signal: np.ndarray, min_gap: int, rel_thresh: float = 0.4) -> list[float]:
    """Find contiguous runs of above-threshold samples; return the center
    index of each run. Threshold = rel_thresh × the signal's max value."""
    if signal.size == 0:
        return []
    max_val = float(signal.max())
    if max_val <= 0:
        return []
    thresh = max_val * rel_thresh
    peaks: list[float] = []
    start: int | None = None
    for i, v in enumerate(signal):
        if v >= thresh:
            if start is None:
                start = i
        elif start is not None:
            end = i - 1
            peaks.append((start + end) / 2.0)
            start = None
    if start is not None:
        peaks.append((start + len(signal) - 1) / 2.0)
    # Merge peaks that are too close (probably double detection of same line).
    if min_gap > 1 and len(peaks) > 1:
        merged = [peaks[0]]
        for p in peaks[1:]:
            if p - merged[-1] < min_gap:
                merged[-1] = (merged[-1] + p) / 2.0
            else:
                merged.append(p)
        peaks = merged
    return [float(p) for p in peaks]


@dataclass(frozen=True)
class SavedStonePoints:
    task_id: str
    black_count: int
    white_count: int


def save_stone_points(
    task_id: str,
    black_points: list[tuple[float, float]],
    white_points: list[tuple[float, float]],
) -> SavedStonePoints:
    """Save stone-center points for a task crop (any source). Coordinates are
    in the crop's pixel space (origin at crop top-left)."""
    STONE_POINTS_DIR.mkdir(parents=True, exist_ok=True)
    png_bytes = get_task_crop(task_id)
    (STONE_POINTS_DIR / f"{task_id}.png").write_bytes(png_bytes)
    (STONE_POINTS_DIR / f"{task_id}.json").write_text(json.dumps({
        "task_id": task_id,
        "black": [[float(x), float(y)] for x, y in black_points],
        "white": [[float(x), float(y)] for x, y in white_points],
    }, indent=2))
    log.info(
        "saved stone points %s: %d black, %d white",
        task_id, len(black_points), len(white_points),
    )
    return SavedStonePoints(
        task_id=task_id,
        black_count=len(black_points),
        white_count=len(white_points),
    )


def count_stone_point_labels() -> dict[str, int]:
    if not STONE_POINTS_DIR.exists():
        return {"labeled_tasks": 0, "black": 0, "white": 0}
    tasks = 0
    black = white = 0
    for jp in STONE_POINTS_DIR.glob("*.json"):
        try:
            d = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        tasks += 1
        black += len(d.get("black", []))
        white += len(d.get("white", []))
    return {"labeled_tasks": tasks, "black": black, "white": white}
