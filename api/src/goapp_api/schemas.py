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
