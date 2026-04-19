import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from . import __version__
from .detection import decode_image, detect_boards, detect_boards_debug
from .inference import ModelNotLoaded, detect_boards_yolo, model_available
from .schemas import (
    BoardBBoxOut,
    BoardCirclesResponse,
    BoardGridResponse,
    DetectBoardsResponse,
    DetectedCircle,
    HealthResponse,
    ListStoneTasksResponse,
    SaveBoardLabelsResponse,
    SaveStonePointsResponse,
    StoneTask,
)
from .training import (
    count_board_labels,
    count_stone_point_labels,
    detect_board_grid,
    detect_stone_circles_for_task,
    get_task_crop,
    ingest_pdf_for_stone_tasks_stream,
    list_stone_tasks,
    save_board_label,
    save_stone_points,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Go Problem Workbook API", version=__version__)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/api/pdf/detect-boards", response_model=DetectBoardsResponse)
async def detect_boards_endpoint(file: UploadFile = File(...)) -> DetectBoardsResponse:
    content = await file.read()
    try:
        img = decode_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Prefer the trained YOLO detector; fall back to the OpenCV heuristic until
    # the model file is available.
    try:
        boards = detect_boards_yolo(img) if model_available() else detect_boards(img)
    except ModelNotLoaded:
        boards = detect_boards(img)
    return DetectBoardsResponse(
        boards=[
            BoardBBoxOut(
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                h_lines=b.h_lines, v_lines=b.v_lines,
            )
            for b in boards
        ]
    )


@app.post("/api/pdf/detect-boards-debug")
async def detect_boards_debug_endpoint(file: UploadFile = File(...)) -> Response:
    content = await file.read()
    try:
        img = decode_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    png = detect_boards_debug(img)
    return Response(content=png, media_type="image/png")


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


@app.get("/api/training/board-grid/{board_id}/{bbox_idx}", response_model=BoardGridResponse)
def board_grid_endpoint(board_id: str, bbox_idx: int) -> BoardGridResponse:
    try:
        grid = detect_board_grid(board_id, bbox_idx)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BoardGridResponse(rows=grid["rows"], cols=grid["cols"])


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


@app.post("/api/training/save-board-labels", response_model=SaveBoardLabelsResponse)
async def save_board_labels_endpoint(
    file: UploadFile = File(...),
    bboxes: str = Form(...),
) -> SaveBoardLabelsResponse:
    try:
        parsed = json.loads(bboxes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid bboxes JSON: {e}") from e
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="bboxes must be a JSON list")
    tuples: list[tuple[int, int, int, int]] = []
    for b in parsed:
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(n, (int, float)) for n in b)):
            raise HTTPException(status_code=400, detail=f"bbox must be [x0,y0,x1,y1]: got {b!r}")
        x0, y0, x1, y1 = (int(n) for n in b)
        if x1 <= x0 or y1 <= y0:
            raise HTTPException(status_code=400, detail=f"invalid bbox: {b}")
        tuples.append((x0, y0, x1, y1))

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    saved = save_board_label(image_bytes, tuples)
    return SaveBoardLabelsResponse(
        label_id=saved.label_id,
        bbox_count=len(tuples),
        total_labels=count_board_labels(),
    )
