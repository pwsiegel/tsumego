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

import cv2
import numpy as np

log = logging.getLogger(__name__)

from .paths import STONE_POINTS_DIR, STONE_TASKS_DIR


def list_stone_tasks() -> list[dict]:
    """Stone-tuning tasks — crops from the most recently ingested PDF."""
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
    """Return PNG bytes for a task crop."""
    png_path = STONE_TASKS_DIR / f"{task_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(task_id)
    return png_path.read_bytes()


def _load_task_crop_array(task_id: str) -> np.ndarray:
    """BGR numpy array for a task crop."""
    png_path = STONE_TASKS_DIR / f"{task_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(task_id)
    arr = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {png_path}")
    return img


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
    from .edge_inference import detect_edges
    from .inference import detect_boards_yolo, model_available

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
            detections = detect_boards_yolo(img_bgr)
        except Exception as e:
            log.warning("detection failed on page %d: %s", page_num + 1, e)
            detections = []
        for b in detections:
            x0 = max(0, b.x0); y0 = max(0, b.y0)
            x1 = min(img_bgr.shape[1], b.x1); y1 = min(img_bgr.shape[0], b.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img_bgr[y0:y1, x0:x1]
            try:
                edges = detect_edges(crop)
            except Exception as e:
                log.warning("edge classifier failed on page %d: %s", page_num + 1, e)
                edges = {"left": False, "right": False, "top": False, "bottom": False}
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
                "confidence": b.confidence,
                "edges": edges,
            }))
            count += 1
        yield {"page": page_num + 1, "tasks_added": count}

    log.info("ingested %d stone tasks from %s", count, source_name)
    yield {"done": True, "total_tasks": count}


def ingest_pdf_for_stone_tasks(pdf_bytes: bytes, source_name: str = "pdf") -> int:
    """Non-streaming wrapper — drains the generator and returns final count."""
    last_count = 0
    for event in ingest_pdf_for_stone_tasks_stream(pdf_bytes, source_name):
        if "total_tasks" in event:
            last_count = event["total_tasks"]
    return last_count


def detect_stone_circles_for_task(
    task_id: str,
    min_r_frac: float = 0.02,
    max_r_frac: float = 0.15,
    hough_param2: int = 40,
    white_ring_thresh: float = 0.1,
) -> list[dict]:
    crop = _load_task_crop_array(task_id)
    return _detect_stone_circles_on_crop(
        crop, min_r_frac=min_r_frac, max_r_frac=max_r_frac,
        hough_param2=hough_param2, white_ring_thresh=white_ring_thresh,
    )


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

    fill_k = max(15, min_r_h * 2 - 5)
    if fill_k % 2 == 0:
        fill_k += 1
    binary_filled = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_k, fill_k)),
    )
    filled_gray = cv2.bitwise_not(binary_filled)
    filled_blurred = cv2.GaussianBlur(filled_gray, (5, 5), 1.5)
    hough_filled = cv2.HoughCircles(
        filled_blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=max(8, int(min_r_h * 1.5)),
        param1=80, param2=hough_param2,
        minRadius=min_r_h, maxRadius=max_r_h,
    )

    hough_all = []
    if hough_raw is not None:
        hough_all.extend(hough_raw[0].tolist())
    if hough_filled is not None:
        hough_all.extend(hough_filled[0].tolist())
    hough = [hough_all] if hough_all else None
    hough_cands: list[dict] = []
    if hough is not None:
        n_samples = 32
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
    out: list[dict] = []
    for s in stones:
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
    """Save stone-center points for a task crop. Coordinates are in the
    crop's pixel space (origin at crop top-left)."""
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
