import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from . import __version__
from .inference import ModelNotLoaded, detect_boards_yolo
from .schemas import (
    BboxDetectResponse,
    BboxUploadResponse,
    BoardBBoxOut,
    BoardDiscretizeLocal,
    BoardListItem,
    BoardListResponse,
    DiscretizedStoneOut,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Go Problem Workbook API", version=__version__)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


# --- bbox tester -----------------------------------------------------------


@app.post("/api/pdf/bbox-upload", response_model=BboxUploadResponse)
async def bbox_upload_endpoint(file: UploadFile = File(...)) -> BboxUploadResponse:
    """Upload a PDF for bbox testing. Wipes any previous PDF, renders all
    pages to disk under BBOX_TEST_DIR, returns the page count."""
    import io
    import shutil
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from .paths import BBOX_TEST_DIR

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    if BBOX_TEST_DIR.exists():
        shutil.rmtree(BBOX_TEST_DIR)
    BBOX_TEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e

    for i, page in enumerate(pdf):
        pil_img = page.render(scale=2.0).to_pil()
        img_bgr = np.array(pil_img)[..., ::-1].copy()
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            continue
        (BBOX_TEST_DIR / f"page_{i:04d}.png").write_bytes(buf.tobytes())

    return BboxUploadResponse(page_count=len(pdf))


@app.get("/api/pdf/bbox-page/{page_idx}.png")
def bbox_page_endpoint(page_idx: int) -> Response:
    from .paths import BBOX_TEST_DIR
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    return Response(content=path.read_bytes(), media_type="image/png")


@app.get("/api/pdf/bbox-detect/{page_idx}", response_model=BboxDetectResponse)
def bbox_detect_endpoint(page_idx: int) -> BboxDetectResponse:
    import cv2
    import numpy as np
    from .paths import BBOX_TEST_DIR
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="could not decode page image")
    h, w = img.shape[:2]
    try:
        boards = detect_boards_yolo(img)
    except ModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return BboxDetectResponse(
        page_index=page_idx,
        page_width=w, page_height=h,
        boards=[
            BoardBBoxOut(
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                confidence=b.confidence,
            ) for b in boards
        ],
    )


# ---------------------------------------------------------------------------
# Per-bbox views of the uploaded PDF. Flat list across all pages, so the
# UI can show one board per screen.
# ---------------------------------------------------------------------------


BOARD_CROP_PAD = 10   # pixels of safety context around each YOLO bbox


def _page_bboxes(page_idx: int):
    """Load the page, run YOLO, return (img, bboxes). Results are cached
    per (path, mtime) so flipping between boards doesn't re-run inference."""
    from .paths import BBOX_TEST_DIR
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
    """Return (crop_bgr, (x0, y0, x1, y1)_in_page) for one detected board.
    The crop is padded by BOARD_CROP_PAD px on each side so edge/pitch
    detection has room to find outer lines that YOLO may have clipped."""
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


@app.get("/api/pdf/boards", response_model=BoardListResponse)
def boards_list_endpoint() -> BoardListResponse:
    """Flat list of every detected board across every page in the currently
    uploaded PDF, in (page, bbox-within-page) order. Drives the UI's
    one-bbox-per-screen navigation."""
    from .paths import BBOX_TEST_DIR
    if not BBOX_TEST_DIR.exists():
        return BoardListResponse(total=0, boards=[])
    pages = sorted(BBOX_TEST_DIR.glob("page_*.png"))
    out: list[BoardListItem] = []
    for page_path in pages:
        try:
            page_idx = int(page_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        _, bboxes = _page_bboxes(page_idx)
        for i, b in enumerate(bboxes):
            out.append(BoardListItem(
                page_idx=page_idx, bbox_idx=i,
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                confidence=b.confidence,
            ))
    return BoardListResponse(total=len(out), boards=out)


@app.get("/api/pdf/board-crop/{page_idx}/{bbox_idx}.png")
def board_crop_endpoint(page_idx: int, bbox_idx: int) -> Response:
    import cv2
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/pdf/board-discretize/{page_idx}/{bbox_idx}",
         response_model=BoardDiscretizeLocal)
def board_discretize_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardDiscretizeLocal:
    """End-to-end 2b pipeline for a single bbox: stone CNN → edge classifier
    → classical discretizer. Returns discrete (col, row) per stone plus the
    inferred grid geometry and 19x19 window placement."""
    from .discretize import discretize
    from .edge_inference import detect_edges
    from .pitch import measure_grid
    from .stone_inference import StoneModelNotLoaded, detect_stones_cnn
    from .stone_inference import model_available as stone_model_available
    if not stone_model_available():
        raise HTTPException(status_code=503, detail="stone detector model not trained")
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    h, w = crop.shape[:2]
    try:
        stones = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    except StoneModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    try:
        edges = detect_edges(crop)
    except Exception:
        edges = {"left": False, "right": False, "top": False, "bottom": False}
    grid = measure_grid(crop, edges)
    pitch = grid["pitch"]
    # Drop detections outside the detected board frame — YOLO likes to
    # fire on circular glyphs in caption text ("problem 36", etc.). Keep
    # a half-pitch margin so stones sitting right on the frame aren't
    # lost.
    # Per-axis pitch selection, in order of reliability:
    # 1. Both opposing frames detected → (R-L)/18 or (B-T)/18 (two-anchor,
    #    very accurate, captures scan anisotropy).
    # 2. One frame on that axis → that side's own per-side pitch (local to
    #    the anchor, better than a global median polluted by stones on
    #    other sides).
    # 3. Neither → global median (or None → stone-based fallback).
    if grid["left"] is not None and grid["right"] is not None:
        pitch_x = (grid["right"] - grid["left"]) / 18
    elif grid["left"] is not None:
        pitch_x = grid["left_pitch"] or pitch
    elif grid["right"] is not None:
        pitch_x = grid["right_pitch"] or pitch
    else:
        pitch_x = pitch
    if grid["top"] is not None and grid["bottom"] is not None:
        pitch_y = (grid["bottom"] - grid["top"]) / 18
    elif grid["top"] is not None:
        pitch_y = grid["top_pitch"] or pitch
    elif grid["bottom"] is not None:
        pitch_y = grid["bottom_pitch"] or pitch
    else:
        pitch_y = pitch
    # Anchor origin to detected frame edges (top/left preferred, else
    # extrapolate from bottom/right assuming a full 19-wide board).
    ox, oy = None, None
    if pitch_y is not None:
        if grid["top"] is not None:
            oy = grid["top"]
        elif grid["bottom"] is not None:
            oy_full = grid["bottom"] - 18 * pitch_y
            oy = oy_full if oy_full >= 0 else grid["bottom"] - int(grid["bottom"] / pitch_y) * pitch_y
    if pitch_x is not None:
        if grid["left"] is not None:
            ox = grid["left"]
        elif grid["right"] is not None:
            ox_full = grid["right"] - 18 * pitch_x
            ox = ox_full if ox_full >= 0 else grid["right"] - int(grid["right"] / pitch_x) * pitch_x
    # Drop detections outside the board area, and drop detections whose
    # bbox is way smaller than a real stone — YOLO fires on:
    #   - caption glyphs ("problem 36", "O", "8", …) → out-of-bounds
    #   - hoshi (star-point dots) → tiny bboxes, inside the board
    if pitch_x is not None and pitch_y is not None and pitch_x > 0 and pitch_y > 0:
        def _lo(frame, origin, pitch):
            if frame is not None:
                return frame - pitch * 0.5
            if origin is not None:
                return origin - pitch * 0.25
            return -1e9

        def _hi(frame, last_near, pitch):
            if frame is not None:
                return frame + pitch * 0.5
            if last_near is not None:
                return last_near + pitch * 0.25
            return 1e9

        top_b = _lo(grid["top"], oy, pitch_y)
        bot_b = _hi(grid["bottom"], grid["top_last"], pitch_y)
        left_b = _lo(grid["left"], ox, pitch_x)
        right_b = _hi(grid["right"], grid["left_last"], pitch_x)
        stones = [
            s for s in stones
            if top_b <= s["y"] <= bot_b and left_b <= s["x"] <= right_b
        ]

        # Drop tiny detections at hoshi intersections (3/9/15, 3/9/15).
        # Hoshi dots are real ink at fixed positions; YOLO reads them
        # as small black stones. Only filter at those positions so we
        # don't accidentally drop a real stone on a board where it just
        # happens to be sitting on a hoshi.
        HOSHI = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                 (15, 3), (15, 9), (15, 15)}
        if ox is not None and oy is not None:
            vc = max(1, min(19, int((w - 1 - ox) / pitch_x) + 1))
            vr = max(1, min(19, int((h - 1 - oy) / pitch_y) + 1))
            # Window placement mirrors discretize._place_window
            cmin = 0 if edges.get("left") else (19 - vc if edges.get("right") else max(0, (19 - vc) // 2))
            rmin = 0 if edges.get("top") else (19 - vr if edges.get("bottom") else max(0, (19 - vr) // 2))
            min_pitch = min(pitch_x, pitch_y)
            # Hoshi detections on real scans come back with r ≈ 0.15-0.25·pitch
            # (YOLO pads a small bbox around the dark dot). Real stones sit at
            # r ≈ 0.4·pitch. A 0.28 threshold cleanly separates them.
            hoshi_r = 0.28 * min_pitch
            def _not_hoshi(s: dict) -> bool:
                cl = max(0, min(vc - 1, int(round((s["x"] - ox) / pitch_x))))
                rl = max(0, min(vr - 1, int(round((s["y"] - oy) / pitch_y))))
                if (cmin + cl, rmin + rl) in HOSHI and s.get("r", 0) < hoshi_r:
                    return False
                return True
            stones = [s for s in stones if _not_hoshi(s)]
    d = discretize(
        stones, w, h, edges=edges,
        cell_size_override=pitch,
        pitch_x_override=pitch_x, pitch_y_override=pitch_y,
        origin_x_override=ox, origin_y_override=oy,
    )
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

