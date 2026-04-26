"""PDF upload, board detection, and discretization endpoints."""

import logging
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from ... import gcs
from ...auth import user_id_from_request
from ...ml.board_detect.detect import ModelNotLoaded, detect_boards_yolo
from ...ml.pipeline import _board_crop, _discretize_board, _page_bboxes
from ...paths import BBOX_TEST_DIR, uploads_dir, uploads_object_key
from .schemas import (
    BboxDetectResponse,
    BboxUploadResponse,
    BoardBBoxOut,
    BoardDiscretizeLocal,
    BoardListItem,
    BoardListResponse,
    IngestFromUploadRequest,
    UploadUrlRequest,
    UploadUrlResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])

UserId = Depends(user_id_from_request)


@router.post("/bbox-upload", response_model=BboxUploadResponse)
async def bbox_upload_endpoint(file: UploadFile = File(...)) -> BboxUploadResponse:
    """Upload a PDF for bbox testing."""
    import io
    import shutil
    import cv2
    import numpy as np
    import pypdfium2 as pdfium

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    if BBOX_TEST_DIR.exists():
        shutil.rmtree(BBOX_TEST_DIR)
    BBOX_TEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e

    for i, page in enumerate(pdf):
        pil_img = page.render(scale=2.0).to_pil()
        img_bgr = np.array(pil_img)[..., ::-1].copy()
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            continue
        (BBOX_TEST_DIR / f"page_{i:04d}.png").write_bytes(buf.tobytes())

    return BboxUploadResponse(page_count=len(pdf))


@router.get("/bbox-page/{page_idx}.png")
def bbox_page_endpoint(page_idx: int) -> Response:
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/bbox-detect/{page_idx}", response_model=BboxDetectResponse)
def bbox_detect_endpoint(page_idx: int) -> BboxDetectResponse:
    import cv2
    import numpy as np
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="could not decode page image")
    h, w = img.shape[:2]
    try:
        boards = detect_boards_yolo(img)
    except ModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return BboxDetectResponse(
        page_index=page_idx,
        page_width=w, page_height=h,
        boards=[
            BoardBBoxOut(
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                confidence=b.confidence,
            ) for b in boards
        ],
    )


@router.get("/boards", response_model=BoardListResponse)
def boards_list_endpoint() -> BoardListResponse:
    """Flat list of every detected board across every page."""
    if not BBOX_TEST_DIR.exists():
        return BoardListResponse(total=0, boards=[])
    pages = sorted(BBOX_TEST_DIR.glob("page_*.png"))
    out: list[BoardListItem] = []
    for page_path in pages:
        try:
            page_idx = int(page_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        _, bboxes = _page_bboxes(page_idx)
        for i, b in enumerate(bboxes):
            out.append(BoardListItem(
                page_idx=page_idx, bbox_idx=i,
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                confidence=b.confidence,
            ))
    return BoardListResponse(total=len(out), boards=out)


@router.get("/board-crop/{page_idx}/{bbox_idx}.png")
def board_crop_endpoint(page_idx: int, bbox_idx: int) -> Response:
    import cv2
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/board-discretize/{page_idx}/{bbox_idx}",
            response_model=BoardDiscretizeLocal)
def board_discretize_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardDiscretizeLocal:
    """End-to-end pipeline for a single bbox."""
    return _discretize_board(page_idx, bbox_idx, peak_thresh)


def _iter_ingest_events(content: bytes, source_name: str, user_id: str) -> Iterator[str]:
    """Render pages, detect boards, save each problem; yield NDJSON events."""
    import io
    import json as _json
    import shutil
    import time
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from ...tsumego import problem_exists, save_problem

    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if BBOX_TEST_DIR.exists():
        shutil.rmtree(BBOX_TEST_DIR)
    BBOX_TEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e
    total_pages = len(pdf)

    yield _json.dumps({
        "event": "start", "source": source_name,
        "uploaded_at": uploaded_at, "total_pages": total_pages,
    }) + "\n"

    for i, page in enumerate(pdf):
        pil_img = page.render(scale=2.0).to_pil()
        img_bgr = np.array(pil_img)[..., ::-1].copy()
        ok, buf = cv2.imencode(".png", img_bgr)
        if ok:
            (BBOX_TEST_DIR / f"page_{i:04d}.png").write_bytes(buf.tobytes())
        yield _json.dumps({
            "event": "page_rendered", "page": i + 1, "total_pages": total_pages,
        }) + "\n"

    total_saved = 0
    skipped = 0
    source_board_idx = 0
    for page_idx in range(total_pages):
        _, bboxes = _page_bboxes(page_idx)
        for bbox_idx in range(len(bboxes)):
            if problem_exists(user_id, source_name, source_board_idx):
                skipped += 1
                source_board_idx += 1
                continue
            try:
                d = _discretize_board(page_idx, bbox_idx)
                crop, _, _ = _board_crop(page_idx, bbox_idx)
                ok, buf = cv2.imencode(".png", crop)
                crop_png = buf.tobytes() if ok else None
                save_problem(
                    user_id=user_id,
                    source=source_name,
                    uploaded_at=uploaded_at,
                    source_board_idx=source_board_idx,
                    stones=[{
                        "col": s.col, "row": s.row, "color": s.color,
                    } for s in d.stones],
                    black_to_play=True,
                    crop_png=crop_png,
                    status="unreviewed",
                    page_idx=page_idx,
                    bbox_idx=bbox_idx,
                )
                total_saved += 1
            except Exception as e:
                log.warning("ingest: board %d (page %d bbox %d) failed: %s",
                            source_board_idx, page_idx, bbox_idx, e)
            yield _json.dumps({
                "event": "board_saved",
                "source_board_idx": source_board_idx,
                "page_idx": page_idx,
                "bbox_idx": bbox_idx,
                "total_saved": total_saved,
            }) + "\n"
            source_board_idx += 1

    yield _json.dumps({
        "event": "done",
        "source": source_name,
        "total_saved": total_saved,
        "skipped": skipped,
    }) + "\n"


@router.post("/ingest")
async def pdf_ingest_endpoint(
    file: UploadFile = File(...), user_id: str = UserId,
) -> StreamingResponse:
    """Multipart ingest. Used locally; in cloud the file is too large to fit
    in a single Cloud Run request, so the client uses the signed-URL flow
    (/upload-url + /ingest-from-upload) instead."""
    content = await file.read()
    source_name = file.filename or "uploaded.pdf"
    return StreamingResponse(
        _iter_ingest_events(content, source_name, user_id),
        media_type="application/x-ndjson",
    )


@router.post("/upload-url", response_model=UploadUrlResponse)
def pdf_upload_url_endpoint(
    request: UploadUrlRequest,
    user_id: str = UserId,
) -> UploadUrlResponse:
    """Tell the client how to deliver the PDF.

    In cloud (GOAPP_GCS_BUCKET set), returns a signed PUT URL targeting the
    bucket directly — bypasses Cloud Run's 32 MiB request-body cap. The
    file lands at a path the FUSE mount can read for ingest. Locally, the
    client falls back to the multipart endpoint."""
    del request  # filename only matters for the eventual ingest call
    if not gcs.is_enabled():
        return UploadUrlResponse(mode="multipart")
    upload_id = uuid.uuid4().hex
    object_key = uploads_object_key(user_id, upload_id)
    url = gcs.signed_upload_url(object_key)
    return UploadUrlResponse(mode="signed-url", upload_id=upload_id, url=url)


@router.post("/ingest-from-upload")
def pdf_ingest_from_upload_endpoint(
    request: IngestFromUploadRequest,
    user_id: str = UserId,
) -> StreamingResponse:
    """Ingest a PDF previously uploaded via signed URL. Reads from the FUSE
    mount path matching the GCS object key, then deletes the file."""
    upload_path = uploads_dir(user_id) / f"{request.upload_id}.pdf"
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="upload not found")
    content = upload_path.read_bytes()

    def iter_events_with_cleanup() -> Iterator[str]:
        try:
            yield from _iter_ingest_events(content, request.filename, user_id)
        finally:
            try:
                upload_path.unlink()
            except FileNotFoundError:
                pass

    return StreamingResponse(
        iter_events_with_cleanup(),
        media_type="application/x-ndjson",
    )
