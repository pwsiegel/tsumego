from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class BoardBBoxOut(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int
    h_lines: int
    v_lines: int


class DetectBoardsResponse(BaseModel):
    boards: list[BoardBBoxOut]


class SaveBoardLabelsResponse(BaseModel):
    label_id: str
    bbox_count: int
    total_labels: int


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


class BoardGridResponse(BaseModel):
    rows: list[float]
    cols: list[float]


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


class PipelineStone(BaseModel):
    col: int
    row: int
    color: str
    x_px: float
    y_px: float


class PipelineResponse(BaseModel):
    stones: list[PipelineStone]
    sgf: str
    edges: dict[str, bool]
    window: dict[str, int]
    pitch: dict[str, float]
    origin: dict[str, float]
    visible_cols: int
    visible_rows: int
