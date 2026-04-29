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


class StoneCenterOut(BaseModel):
    x: float
    y: float
    color: str


class FusedLatticeOut(BaseModel):
    """Fused-lattice fit result. Either pitch may be None when the fit
    couldn't fix that axis; the frontend draws what's available."""
    pitch_x: float | None
    pitch_y: float | None
    origin_x: float | None
    origin_y: float | None
    edges: dict[str, bool]


class SegmentOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class JunctionOut(BaseModel):
    x: float
    y: float
    kind: str  # "T" | "L" | "+" | "I" | "?"
    arms: int  # bitmask N=1, E=2, S=4, W=8
    outward: list[str]  # subset of "N", "E", "S", "W"


class BoardIntersections(BaseModel):
    """Signals consumed by `_discretize_board`, exposed to the dev tool.

    All fields are restricted to what the discretization pipeline
    actually uses: stones (after grid-bbox filter), skeleton junctions
    and edges, segments (after grid-bbox filter) and the fused lattice
    fit derived from them.
    """
    page_idx: int
    bbox_idx: int
    crop_width: int
    crop_height: int
    stones: list[StoneCenterOut]
    segments: list[SegmentOut]
    fused_lattice: FusedLatticeOut | None
    skeleton_junctions: list[JunctionOut]


class SideTallyOut(BaseModel):
    t: int
    l: int
    total: int


class StoneEdgeClassOut(BaseModel):
    x: float
    y: float
    r: float
    color: str  # "B" | "W"
    sides: dict[str, bool]  # keys "N", "E", "S", "W"


class BoardTJunctionEdges(BaseModel):
    """Segment-topology edge detection on one bbox (dev tool)."""
    page_idx: int
    bbox_idx: int
    crop_width: int
    crop_height: int
    segments: list[SegmentOut]
    junctions: list[JunctionOut]
    sides: dict[str, SideTallyOut]
    edges: dict[str, bool]
    stone_edges: list[StoneEdgeClassOut]


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


class IngestJobOut(BaseModel):
    job_id: str
    source: str
    phase: Literal["rendering", "detecting", "done", "error"]
    started_at: str
    updated_at: str
    total_pages: int | None
    pages_rendered: int
    pages_detected: int
    total_saved: int
    skipped: int
    error: str | None
    stalled: bool


class IngestJobsResponse(BaseModel):
    jobs: list[IngestJobOut]


class IngestJobStartResponse(BaseModel):
    job_id: str


# --- patch sessions: interactive ingest with skip/add over detected bboxes ---


class PatchSessionStartResponse(BaseModel):
    session_id: str


class PatchBBoxOut(BaseModel):
    bbox_idx: int
    x0: int
    y0: int
    x1: int
    y1: int


class PatchPageOut(BaseModel):
    page_idx: int
    image_w: int
    image_h: int
    bboxes: list[PatchBBoxOut]


class PatchApplyProgress(BaseModel):
    total: int
    ingested: int
    failed: int


class PatchSessionOut(BaseModel):
    session_id: str
    source: str
    phase: Literal["rendering", "detecting", "ready", "applying", "done", "error"]
    started_at: str
    updated_at: str
    total_pages: int | None
    pages_rendered: int
    pages_detected: int
    pages: list[PatchPageOut]
    apply: PatchApplyProgress | None
    error: str | None


class PatchSessionsResponse(BaseModel):
    sessions: list[PatchSessionOut]


class PatchSkipBBox(BaseModel):
    page_idx: int
    bbox_idx: int


class PatchAddBBox(BaseModel):
    page_idx: int
    x0: int
    y0: int
    x1: int
    y1: int


class PatchApplyRequest(BaseModel):
    skip: list[PatchSkipBBox] = []
    adds: list[PatchAddBBox] = []


class PatchApplyAck(BaseModel):
    """Apply runs in the background — the client polls
    `/patch-sessions/{id}` (or the list endpoint) for progress."""
    session_id: str
