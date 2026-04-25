from pydantic import BaseModel


class TsumegoStone(BaseModel):
    col: int
    row: int
    color: str              # "B" or "W"


class SaveTsumegoRequest(BaseModel):
    source: str
    uploaded_at: str
    source_board_idx: int
    page_idx: int
    bbox_idx: int
    stones: list[TsumegoStone]
    black_to_play: bool = True
    status: str = "accepted"


class SaveTsumegoResponse(BaseModel):
    id: str


class Collection(BaseModel):
    source: str
    count: int
    accepted: int = 0
    accepted_edited: int = 0
    rejected: int = 0
    unreviewed: int = 0
    last_uploaded_at: str


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
