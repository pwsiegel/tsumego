import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from . import __version__
from .inference import ModelNotLoaded, detect_boards_yolo
from .schemas import (
    BboxDetectResponse,
    BboxUploadResponse,
    BoardBBoxOut,
    BoardCirclesResponse,
    BoardListItem,
    BoardListResponse,
    BoardStonesLocal,
    GridDetectResponse,
    ClearStoneTasksResponse,
    CnnStone,
    CnnStonesResponse,
    DetectedCircle,
    EdgeProbsResponse,
    GridResponse,
    HealthResponse,
    ListStoneTasksResponse,
    SaveStonePointsResponse,
    SgfResponse,
    SgfStone,
    StoneTask,
)
from .grid_inference import (
    GridModelNotLoaded,
    model_available as grid_model_available,
    predict_grid,
)
from .grid_to_sgf import grid_to_sgf, window_from_edges_and_bbox
from .edge_inference import (
    EdgeModelNotLoaded,
    detect_edges_with_probs,
    model_available as edge_model_available,
)
from .sgf import stones_to_sgf
from .stone_inference import (
    StoneModelNotLoaded,
    detect_stones_cnn,
)
from .stone_inference import model_available as stone_model_available
from .training import (
    _load_task_crop_array,
    count_stone_point_labels,
    detect_stone_circles_for_task,
    get_task_crop,
    ingest_pdf_for_stone_tasks_stream,
    list_stone_tasks,
    save_stone_points,
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
    The crop is padded by BOARD_CROP_PAD px on each side so grid_detect has
    room to find outer lines that YOLO may have clipped."""
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


@app.get("/api/pdf/board-stones/{page_idx}/{bbox_idx}", response_model=BoardStonesLocal)
def board_stones_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardStonesLocal:
    """Run the stone CNN on a bbox crop and return stone centers + colors
    in crop-local pixel coordinates."""
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
    return BoardStonesLocal(
        page_idx=page_idx, bbox_idx=bbox_idx,
        crop_width=w, crop_height=h,
        stones=[CnnStone(**s) for s in stones],
    )


@app.get("/api/training/stone-tasks", response_model=ListStoneTasksResponse)
def list_stone_tasks_endpoint() -> ListStoneTasksResponse:
    tasks = list_stone_tasks()
    return ListStoneTasksResponse(
        tasks=[StoneTask(**t) for t in tasks],
        totals=count_stone_point_labels(),
    )


@app.get("/api/training/task-crops/{task_id}.png")
def task_crop_endpoint(task_id: str) -> Response:
    try:
        png = get_task_crop(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(content=png, media_type="image/png")


@app.post("/api/training/ingest-pdf-for-stones")
async def ingest_pdf_for_stones_endpoint(file: UploadFile = File(...)) -> Response:
    """Stream NDJSON progress events as the PDF is processed."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    source_name = file.filename or "pdf"

    def iter_events():
        try:
            for ev in ingest_pdf_for_stone_tasks_stream(content, source_name):
                yield json.dumps(ev) + "\n"
        except RuntimeError as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


@app.get("/api/training/task-circles/{task_id}", response_model=BoardCirclesResponse)
def task_circles_endpoint(
    task_id: str,
    min_r_frac: float = 0.02,
    max_r_frac: float = 0.15,
    hough_param2: int = 40,
    white_ring_thresh: float = 0.1,
) -> BoardCirclesResponse:
    try:
        circles = detect_stone_circles_for_task(
            task_id,
            min_r_frac=min_r_frac, max_r_frac=max_r_frac,
            hough_param2=hough_param2, white_ring_thresh=white_ring_thresh,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BoardCirclesResponse(circles=[DetectedCircle(**c) for c in circles])


@app.get("/api/training/task-stones-cnn/{task_id}", response_model=CnnStonesResponse)
def task_stones_cnn_endpoint(
    task_id: str,
    peak_thresh: float = 0.3,
) -> CnnStonesResponse:
    if not stone_model_available():
        raise HTTPException(status_code=503, detail="stone detector model not trained")
    try:
        crop = _load_task_crop_array(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        stones = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    except StoneModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return CnnStonesResponse(stones=[CnnStone(**s) for s in stones])


@app.get("/api/training/task-edges/{task_id}", response_model=EdgeProbsResponse)
def task_edges_endpoint(task_id: str) -> EdgeProbsResponse:
    if not edge_model_available():
        raise HTTPException(status_code=503, detail="edge classifier not trained")
    try:
        crop = _load_task_crop_array(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        probs = detect_edges_with_probs(crop)
    except EdgeModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return EdgeProbsResponse(**probs)


@app.get("/api/training/task-grid-detect/{task_id}", response_model=GridDetectResponse)
def task_grid_detect_endpoint(task_id: str) -> GridDetectResponse:
    """Classical Hough-based grid detector: returns pitch/origin in the
    original crop's pixel space plus the raw detected line positions for
    overlay."""
    from .edge_inference import detect_edges
    from .grid_detect import detect_grid
    try:
        crop = _load_task_crop_array(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    h, w = crop.shape[:2]
    result = detect_grid(crop)
    try:
        edges = detect_edges(crop)
    except Exception:
        edges = {"left": False, "right": False, "top": False, "bottom": False}
    return GridDetectResponse(
        crop_width=w, crop_height=h,
        pitch_x_px=result.pitch_x_px, pitch_y_px=result.pitch_y_px,
        origin_x_px=result.origin_x_px, origin_y_px=result.origin_y_px,
        vert_xs=result.vert_xs, horz_ys=result.horz_ys,
        edges=edges,
    )


@app.get("/api/training/task-grid/{task_id}", response_model=GridResponse)
def task_grid_endpoint(task_id: str) -> GridResponse:
    """Run the grid classifier on this task's crop and return the 19x19
    prediction + implied window + SGF. The crop is the YOLO-produced bbox
    (tight-ish since YOLO was trained on tight bboxes); the edges attached
    to the task at ingest time tell us which sides are real boundaries."""
    if not grid_model_available():
        raise HTTPException(status_code=503, detail="grid classifier not trained")
    import json
    from .training import STONE_TASKS_DIR
    json_path = STONE_TASKS_DIR / f"{task_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=task_id)
    meta = json.loads(json_path.read_text())
    edges = meta.get("edges", {"left": False, "right": False, "top": False, "bottom": False})
    try:
        crop = _load_task_crop_array(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    try:
        grid, probs = predict_grid(crop)
    except GridModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Figure out which window this crop corresponds to. When L+R both True
    # or T+B both True we know the full extent (0..18). Otherwise we look at
    # the model's predictions: the visible window should be the smallest
    # contiguous region around the stones it predicted.
    stone_rows = [r for r in range(19) for c in range(19) if grid[r, c] > 0]
    stone_cols = [c for c in range(19) for r in range(19) if grid[r, c] > 0]
    n_cells_w = max(6, (max(stone_cols) - min(stone_cols) + 1) if stone_cols else 10)
    n_cells_h = max(6, (max(stone_rows) - min(stone_rows) + 1) if stone_rows else 10)
    window = window_from_edges_and_bbox(edges, n_cells_w, n_cells_h)
    sgf = grid_to_sgf(grid, window)
    return GridResponse(
        grid=grid.tolist(),
        edges=edges,
        window=window,
        sgf=sgf,
    )


@app.post("/api/training/clear-stone-tasks", response_model=ClearStoneTasksResponse)
def clear_stone_tasks_endpoint() -> ClearStoneTasksResponse:
    import shutil
    from .training import STONE_TASKS_DIR
    if not STONE_TASKS_DIR.exists():
        return ClearStoneTasksResponse(removed=0)
    removed = sum(1 for _ in STONE_TASKS_DIR.glob("*.png"))
    shutil.rmtree(STONE_TASKS_DIR)
    STONE_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    return ClearStoneTasksResponse(removed=removed)


@app.get("/api/training/task-sgf/{task_id}", response_model=SgfResponse)
def task_sgf_endpoint(
    task_id: str,
    peak_thresh: float = 0.3,
) -> SgfResponse:
    if not stone_model_available():
        raise HTTPException(status_code=503, detail="stone detector model not trained")
    try:
        crop = _load_task_crop_array(task_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        detected = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    except StoneModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    sgf_text, mapping = stones_to_sgf(crop, detected)
    placed: list[SgfStone] = []
    for s in detected:
        col = int(round((s["x"] - mapping.origin_x) / max(mapping.pitch, 1e-6)))
        row = int(round((s["y"] - mapping.origin_y) / max(mapping.pitch, 1e-6)))
        col = max(0, min(18, col))
        row = max(0, min(18, row))
        placed.append(SgfStone(col=col, row=row, color=s["color"]))
    return SgfResponse(
        sgf=sgf_text,
        stones=placed,
        pitch=mapping.pitch,
        origin_x=mapping.origin_x,
        origin_y=mapping.origin_y,
        edges_detected=mapping.edges_detected,
    )


@app.post("/api/training/save-stone-points", response_model=SaveStonePointsResponse)
def save_stone_points_endpoint(
    task_id: str = Form(...),
    black: str = Form(...),
    white: str = Form(...),
) -> SaveStonePointsResponse:
    try:
        black_pts = [tuple(p) for p in json.loads(black)]
        white_pts = [tuple(p) for p in json.loads(white)]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid points JSON: {e}") from e
    try:
        saved = save_stone_points(task_id, black_pts, white_pts)
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return SaveStonePointsResponse(
        task_id=saved.task_id,
        black_count=saved.black_count,
        white_count=saved.white_count,
        totals=count_stone_point_labels(),
    )


