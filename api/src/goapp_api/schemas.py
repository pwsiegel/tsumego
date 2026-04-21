from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class BoardBBoxOut(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


class StoneTask(BaseModel):
    task_id: str
    source: str
    labeled: bool


class ListStoneTasksResponse(BaseModel):
    tasks: list[StoneTask]
    totals: dict[str, int]


class IngestPdfResponse(BaseModel):
    tasks_added: int
    total_tasks: int


class SaveStonePointsResponse(BaseModel):
    task_id: str
    black_count: int
    white_count: int
    totals: dict[str, int]


class DetectedCircle(BaseModel):
    x: float
    y: float
    r: float
    color: str | None = None  # 'B' or 'W' when auto-classified


class BoardCirclesResponse(BaseModel):
    circles: list[DetectedCircle]


class CnnStone(BaseModel):
    x: float
    y: float
    r: float
    color: str
    conf: float


class CnnStonesResponse(BaseModel):
    stones: list[CnnStone]


class SgfStone(BaseModel):
    col: int
    row: int
    color: str


class SgfResponse(BaseModel):
    sgf: str
    stones: list[SgfStone]
    pitch: float
    origin_x: float
    origin_y: float
    edges_detected: dict[str, bool]


class EdgeProbsResponse(BaseModel):
    left: float
    right: float
    top: float
    bottom: float


class ClearStoneTasksResponse(BaseModel):
    removed: int


class GridResponse(BaseModel):
    grid: list[list[int]]            # 19x19 of 0=empty, 1=B, 2=W
    edges: dict[str, bool]           # board-boundary bits from YOLO
    window: dict[str, int]           # col_min, col_max, row_min, row_max
    sgf: str                         # SGF built from grid + window


class GridDetectResponse(BaseModel):
    crop_width: int
    crop_height: int
    pitch_x_px: float | None
    pitch_y_px: float | None
    origin_x_px: float | None
    origin_y_px: float | None
    vert_xs: list[float]    # detected vertical-line x-positions (pixels)
    horz_ys: list[float]    # detected horizontal-line y-positions (pixels)
    edges: dict[str, bool]


class BoardListItem(BaseModel):
    page_idx: int
    bbox_idx: int                    # position within this page's bbox list
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


class BoardListResponse(BaseModel):
    total: int
    boards: list[BoardListItem]


class BoardStonesLocal(BaseModel):
    """Stone detections for a single bbox crop, in CROP-LOCAL pixel
    coordinates (origin = top-left of the padded crop returned by
    /api/pdf/board-crop/{page_idx}/{bbox_idx}.png)."""
    page_idx: int
    bbox_idx: int
    crop_width: int
    crop_height: int
    stones: list[CnnStone]


class BboxUploadResponse(BaseModel):
    page_count: int


class BboxDetectResponse(BaseModel):
    page_index: int
    page_width: int
    page_height: int
    boards: list[BoardBBoxOut]
