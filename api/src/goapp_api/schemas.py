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


class TsumegoStone(BaseModel):
    col: int
    row: int
    color: str              # "B" or "W"


class SaveTsumegoRequest(BaseModel):
    source: str             # PDF file name the problem came from
    uploaded_at: str        # ISO 8601 timestamp set by the client at upload time
    source_board_idx: int   # flat index across the uploaded PDF's detected boards
    page_idx: int           # page in the uploaded PDF this crop came from
    bbox_idx: int           # bbox index within that page
    stones: list[TsumegoStone]
    black_to_play: bool = True
    status: str = "accepted"  # "accepted" | "rejected"


class SaveTsumegoResponse(BaseModel):
    id: str


class Collection(BaseModel):
    source: str                 # source PDF file name
    count: int                  # total problems in the collection
    accepted: int = 0
    accepted_edited: int = 0
    rejected: int = 0
    unreviewed: int = 0
    last_uploaded_at: str       # ISO 8601 timestamp (most recent upload)


class CollectionsResponse(BaseModel):
    collections: list[Collection]


class DeleteCollectionResponse(BaseModel):
    source: str
    removed: int


class TsumegoProblem(BaseModel):
    id: str
    source: str
    uploaded_at: str
    source_board_idx: int
    page_idx: int | None = None
    bbox_idx: int | None = None
    status: str
    image: str | None
    black_to_play: bool
    stones: list[TsumegoStone]


class ProblemsResponse(BaseModel):
    problems: list[TsumegoProblem]


class UpdateProblemRequest(BaseModel):
    status: str | None = None
    stones: list[TsumegoStone] | None = None
    black_to_play: bool | None = None


class UpdateProblemResponse(BaseModel):
    id: str
    status: str
