"""Tsumego collection and problem management endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from ...auth import user_id_from_request
from ...ml.pipeline import _board_crop
from ...paths import tsumego_dir
from ...tsumego import (
    delete_collection,
    delete_problem,
    list_collections,
    list_problems,
    load_problem,
    rename_collection,
    save_problem,
    update_problem,
)
from .schemas import (
    Collection,
    CollectionsResponse,
    DeleteCollectionResponse,
    ProblemsResponse,
    RenameCollectionRequest,
    RenameCollectionResponse,
    SaveTsumegoRequest,
    SaveTsumegoResponse,
    TsumegoProblem,
    TsumegoStone,
    UpdateProblemRequest,
    UpdateProblemResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["tsumego"])

UserId = Depends(user_id_from_request)


@router.post("/api/tsumego/save", response_model=SaveTsumegoResponse)
def tsumego_save_endpoint(
    req: SaveTsumegoRequest, user_id: str = UserId,
) -> SaveTsumegoResponse:
    """Persist an accepted problem as SGF + metadata JSON + crop PNG."""
    import cv2

    crop_png: bytes | None = None
    try:
        crop, _, _ = _board_crop(req.page_idx, req.bbox_idx)
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            crop_png = buf.tobytes()
    except HTTPException:
        pass

    stones = [s.model_dump() for s in req.stones]
    path = save_problem(
        user_id=user_id,
        source=req.source,
        uploaded_at=req.uploaded_at,
        source_board_idx=req.source_board_idx,
        stones=stones,
        black_to_play=req.black_to_play,
        crop_png=crop_png,
        status=req.status,
    )
    return SaveTsumegoResponse(id=path.stem)


@router.get("/api/tsumego/collections", response_model=CollectionsResponse)
def tsumego_collections_endpoint(user_id: str = UserId) -> CollectionsResponse:
    items = list_collections(user_id)
    return CollectionsResponse(
        collections=[Collection(**c) for c in items]
    )


@router.post("/api/tsumego/collections/{source:path}/rename",
             response_model=RenameCollectionResponse)
def tsumego_collection_rename_endpoint(
    source: str, req: RenameCollectionRequest, user_id: str = UserId,
) -> RenameCollectionResponse:
    try:
        renamed = rename_collection(user_id, source, req.new_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RenameCollectionResponse(
        old_source=source, new_source=req.new_source.strip(), renamed=renamed,
    )


@router.delete("/api/tsumego/collections/{source:path}",
               response_model=DeleteCollectionResponse)
def tsumego_collection_delete_endpoint(
    source: str, user_id: str = UserId,
) -> DeleteCollectionResponse:
    removed = delete_collection(user_id, source)
    return DeleteCollectionResponse(source=source, removed=removed)


@router.get("/api/tsumego/collections/{source:path}/problems",
            response_model=ProblemsResponse)
def tsumego_collection_problems_endpoint(
    source: str, user_id: str = UserId,
) -> ProblemsResponse:
    items = list_problems(user_id, source)
    return ProblemsResponse(problems=[
        TsumegoProblem(
            id=p["id"], source=p["source"], uploaded_at=p["uploaded_at"],
            source_board_idx=p["source_board_idx"],
            page_idx=p.get("page_idx"), bbox_idx=p.get("bbox_idx"),
            status=p.get("status", "unreviewed"),
            image=p.get("image"),
            black_to_play=p.get("black_to_play", True),
            stones=[TsumegoStone(**s) for s in p.get("stones", [])],
        ) for p in items
    ])


@router.get("/api/tsumego/{problem_id}/image.png")
def tsumego_image_endpoint(problem_id: str, user_id: str = UserId) -> Response:
    meta = load_problem(user_id, problem_id)
    if meta is None or not meta.get("image"):
        raise HTTPException(status_code=404, detail="image not found")
    path = tsumego_dir(user_id) / meta["image"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/api/tsumego/{problem_id}", response_model=TsumegoProblem)
def tsumego_get_endpoint(problem_id: str, user_id: str = UserId) -> TsumegoProblem:
    p = load_problem(user_id, problem_id)
    if p is None:
        raise HTTPException(status_code=404, detail=problem_id)
    return TsumegoProblem(
        id=p["id"], source=p["source"], uploaded_at=p["uploaded_at"],
        source_board_idx=p["source_board_idx"],
        page_idx=p.get("page_idx"), bbox_idx=p.get("bbox_idx"),
        status=p.get("status", "unreviewed"),
        image=p.get("image"),
        black_to_play=p.get("black_to_play", True),
        stones=[TsumegoStone(**s) for s in p.get("stones", [])],
    )


@router.delete("/api/tsumego/{problem_id}")
def tsumego_delete_endpoint(problem_id: str, user_id: str = UserId) -> dict:
    if not delete_problem(user_id, problem_id):
        raise HTTPException(status_code=404, detail=problem_id)
    return {"id": problem_id, "deleted": True}


@router.post("/api/tsumego/{problem_id}/status", response_model=UpdateProblemResponse)
def tsumego_update_endpoint(
    problem_id: str, req: UpdateProblemRequest, user_id: str = UserId,
) -> UpdateProblemResponse:
    stones = [s.model_dump() for s in req.stones] if req.stones is not None else None
    try:
        meta = update_problem(
            user_id,
            problem_id,
            status=req.status,
            stones=stones,
            black_to_play=req.black_to_play,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if meta is None:
        raise HTTPException(status_code=404, detail=problem_id)
    return UpdateProblemResponse(id=meta["id"], status=meta["status"])
