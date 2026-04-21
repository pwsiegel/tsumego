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


class BboxUploadResponse(BaseModel):
    page_count: int


class BboxDetectResponse(BaseModel):
    page_index: int
    page_width: int
    page_height: int
    boards: list[BoardBBoxOut]


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
    """Full pipeline output for a single bbox: stones + their snapped
    discrete positions + inferred grid geometry + window placement."""
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
