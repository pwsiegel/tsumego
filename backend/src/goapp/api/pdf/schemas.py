from typing import Literal

from pydantic import BaseModel


class BoardBBoxOut(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


class BboxUploadResponse(BaseModel):
    page_count: int


class BboxDetectResponse(BaseModel):
    page_index: int
    page_width: int
    page_height: int
    boards: list[BoardBBoxOut]


class BoardListItem(BaseModel):
    page_idx: int
    bbox_idx: int
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


class BoardListResponse(BaseModel):
    total: int
    boards: list[BoardListItem]


class DiscretizedStoneOut(BaseModel):
    x: float
    y: float
    color: str
    conf: float
    col_local: int
    row_local: int
    col: int
    row: int


class BoardDiscretizeLocal(BaseModel):
    """Full pipeline output for a single bbox."""
    page_idx: int
    bbox_idx: int
    crop_width: int
    crop_height: int
    cell_size: float
    origin_x: float
    origin_y: float
    visible_cols: int
    visible_rows: int
    col_min: int
    row_min: int
    edges: dict[str, bool]
    stones: list[DiscretizedStoneOut]


class BoardGridDetect(BaseModel):
    """Raw output of the grid-geometry regressor for one bbox."""
    page_idx: int
    bbox_idx: int
    crop_width: int
    crop_height: int
    grid_x0: float
    grid_y0: float
    grid_x1: float
    grid_y1: float
    pitch_x: float
    pitch_y: float
    edges: dict[str, bool]


class UploadUrlRequest(BaseModel):
    filename: str


class UploadUrlResponse(BaseModel):
    """Where the client should send the PDF.

    `signed-url`: PUT the file to `url` with Content-Type `application/pdf`,
    then call `/api/pdf/ingest-from-upload` with `upload_id`.
    `multipart`: POST the file to `/api/pdf/ingest` (legacy, local-only).
    """
    mode: Literal["signed-url", "multipart"]
    upload_id: str | None = None
    url: str | None = None


class IngestFromUploadRequest(BaseModel):
    upload_id: str
    filename: str
