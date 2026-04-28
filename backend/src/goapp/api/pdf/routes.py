"""PDF upload, board detection, and discretization endpoints."""

import json as _json
import logging
import threading
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from ... import gcs, ingest_jobs
from ...auth import user_id_from_request
from ...ml.pipeline import (
    _board_crop,
    _discretize_board,
    _page_bboxes,
    _page_bboxes_cached,
)
from ...ml.stone_detect.clean import paint_out_stones
from ...ml.stone_detect.detect import detect_stones_cnn
from ...paths import (
    BBOX_TEST_DIR,
    uploads_dir,
    uploads_object_key,
)
from .schemas import (
    BboxDetectResponse,
    BboxUploadResponse,
    BoardBBoxOut,
    BoardDiscretizeLocal,
    BoardIntersections,
    BoardListItem,
    BoardListResponse,
    BoardTJunctionEdges,
    FusedLatticeOut,
    IngestFromUploadRequest,
    IngestJobOut,
    IngestJobsResponse,
    IngestJobStartResponse,
    JunctionOut,
    SegmentOut,
    SideTallyOut,
    StoneCenterOut,
    StoneEdgeClassOut,
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
    _page_bboxes_cached.cache_clear()

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


@router.post("/bbox-ingest-stream")
async def bbox_ingest_stream_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload + render + detect in one streaming pass.

    Wire format (NDJSON, one event per line):
      {"event":"start","stage":"render","total":N}
      {"event":"page-rendered","index":i}            # one per page
      {"event":"start","stage":"detect","total":N}
      {"event":"page-detected","index":i,"boards":[...]}  # one per page
      {"event":"done","page_count":N,"board_count":M}

    Each `boards` payload is a list of BoardListItem dicts so the web
    view can build the full boards list incrementally without a
    separate /boards round-trip.
    """
    import io
    import json as _json
    import shutil
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from ...ml.board_detect.detect import ModelNotLoaded

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    if BBOX_TEST_DIR.exists():
        shutil.rmtree(BBOX_TEST_DIR)
    BBOX_TEST_DIR.mkdir(parents=True, exist_ok=True)
    _page_bboxes_cached.cache_clear()

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e
    n_pages = len(pdf)

    def generate():
        yield _json.dumps({"event": "start", "stage": "render", "total": n_pages}) + "\n"
        for i in range(n_pages):
            pil_img = pdf[i].render(scale=2.0).to_pil()
            img_bgr = np.array(pil_img)[..., ::-1].copy()
            ok, buf = cv2.imencode(".png", img_bgr)
            if ok:
                (BBOX_TEST_DIR / f"page_{i:04d}.png").write_bytes(buf.tobytes())
            yield _json.dumps({"event": "page-rendered", "index": i}) + "\n"

        yield _json.dumps({"event": "start", "stage": "detect", "total": n_pages}) + "\n"
        board_count = 0
        for i in range(n_pages):
            page_path = BBOX_TEST_DIR / f"page_{i:04d}.png"
            if not page_path.exists():
                yield _json.dumps({"event": "page-detected", "index": i, "boards": []}) + "\n"
                continue
            try:
                bboxes = _page_bboxes_cached(str(page_path), page_path.stat().st_mtime_ns)
            except ModelNotLoaded as e:
                yield _json.dumps({"event": "error", "message": str(e)}) + "\n"
                return
            page_boards = [{
                "page_idx": i, "bbox_idx": j,
                "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
                "confidence": b.confidence,
            } for j, b in enumerate(bboxes)]
            board_count += len(page_boards)
            yield _json.dumps({"event": "page-detected", "index": i, "boards": page_boards}) + "\n"

        yield _json.dumps({
            "event": "done",
            "page_count": n_pages,
            "board_count": board_count,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/bbox-page/{page_idx}.png")
def bbox_page_endpoint(page_idx: int) -> Response:
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/bbox-detect/{page_idx}", response_model=BboxDetectResponse)
def bbox_detect_endpoint(page_idx: int) -> BboxDetectResponse:
    """Return raw YOLO bboxes for the page."""
    img, raw = _page_bboxes(page_idx)
    h, w = img.shape[:2]
    return BboxDetectResponse(
        page_index=page_idx,
        page_width=w, page_height=h,
        boards=[
            BoardBBoxOut(
                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                confidence=b.confidence,
            ) for b in raw
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


@router.get("/board-cleaned/{page_idx}/{bbox_idx}.png")
def board_cleaned_endpoint(
    page_idx: int, bbox_idx: int, peak_thresh: float = 0.3,
) -> Response:
    """Crop with detected stones painted out (board color filled in).

    Used by the dev IntersectionTest view to compare classical geometry on the
    raw vs cleaned crop."""
    import cv2
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    stone_dets = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    cleaned = paint_out_stones(crop, stone_dets)
    ok, buf = cv2.imencode(".png", cleaned)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/board-skeleton/{page_idx}/{bbox_idx}.png")
def board_skeleton_endpoint(
    page_idx: int, bbox_idx: int, peak_thresh: float = 0.3,
) -> Response:
    """Render the actual skeleton the edge detector consumes.

    Stones are painted out, then `_skeletonize` (adaptive threshold +
    main-CC bbox filter + skimage.skeletonize) produces a 1-px-wide
    skeleton. This endpoint returns that skeleton as a black-on-white
    PNG so the dev tool can swap it in place of the cleaned crop.
    """
    import cv2
    import numpy as np
    from ...ml.edge_detect.tjunction import _skeletonize
    from ...ml.stone_detect.clean import paint_out_stones
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    stone_dets = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    cleaned = paint_out_stones(crop, stone_dets)
    skel = _skeletonize(cleaned)
    img = np.where(skel, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
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


@router.get("/board-intersections/{page_idx}/{bbox_idx}",
            response_model=BoardIntersections)
def board_intersections_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardIntersections:
    """Mirror of the signals `_discretize_board` consumes.

    Stones (after grid-bbox filter), skeleton edges + junctions, and the
    fused lattice fit from segments + stones + skeleton junctions inside
    the grid bbox. Anything outside that path is intentionally omitted —
    this endpoint exists so the dev tool shows exactly what
    discretization sees.
    """
    from ...ml.edge_detect.skeleton import decide_edges
    from ...ml.edge_detect.tjunction import main_grid_bbox
    from ...ml.segments.detect import detect_segments
    from ...ml.segments.reason import fit_lattice_fused
    from ...ml.stone_detect.clean import filter_to_grid_bbox

    crop, _, _ = _board_crop(page_idx, bbox_idx)
    h, w = crop.shape[:2]
    stone_dets = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    gbb = main_grid_bbox(crop)
    stones = filter_to_grid_bbox(stone_dets, gbb)
    try:
        skel_result = decide_edges(crop, stones)
        skel_edges = skel_result.edges
        skel_junctions_raw = skel_result.junctions
    except Exception:
        skel_edges = {"left": False, "right": False, "top": False, "bottom": False}
        skel_junctions_raw = []

    cleaned_crop = paint_out_stones(crop, stones)
    segs = detect_segments(cleaned_crop)

    if gbb is not None:
        gx0, gy0, gx1, gy1 = gbb

        def _seg_in_gbb(s) -> bool:
            mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
            return gx0 <= mx <= gx1 and gy0 <= my <= gy1

        segs_for_fit = [s for s in segs if _seg_in_gbb(s)]
        ix_for_fit = [(j.x, j.y) for j in skel_junctions_raw
                      if gx0 <= j.x <= gx1 and gy0 <= j.y <= gy1]
    else:
        segs_for_fit = segs
        ix_for_fit = [(j.x, j.y) for j in skel_junctions_raw]

    fused = fit_lattice_fused(
        segs_for_fit,
        stone_centers=[(s["x"], s["y"]) for s in stones],
        intersection_centers=ix_for_fit,
        crop_w=w, crop_h=h,
        stone_radii=[s["r"] for s in stones if s.get("r")],
    )
    fused_lattice_out = FusedLatticeOut(
        pitch_x=fused.x.pitch, pitch_y=fused.y.pitch,
        origin_x=fused.x.origin, origin_y=fused.y.origin,
        edges=skel_edges,
    )
    return BoardIntersections(
        page_idx=page_idx, bbox_idx=bbox_idx,
        crop_width=w, crop_height=h,
        stones=[
            StoneCenterOut(x=d["x"], y=d["y"], color=d["color"])
            for d in stones
        ],
        segments=[
            SegmentOut(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2)
            for s in segs_for_fit
        ],
        fused_lattice=fused_lattice_out,
        skeleton_junctions=[
            JunctionOut(x=j.x, y=j.y, kind=j.kind, arms=j.arms, outward=list(j.outward))
            for j in skel_junctions_raw
        ],
    )


@router.get("/board-tjunctions/{page_idx}/{bbox_idx}",
            response_model=BoardTJunctionEdges)
def board_tjunctions_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardTJunctionEdges:
    """Skeleton-topology edge detection (dev tool).

    Pipeline: paint stones out → binarize + skeletonize → count 8-
    neighbors per skeleton pixel → cluster junction pixels and recover
    arm directions by walking the skeleton outward → tally per side."""
    from ...ml.edge_detect.skeleton import decide_edges
    from ...ml.edge_detect.tjunction import main_grid_bbox
    from ...ml.stone_detect.clean import filter_to_grid_bbox

    crop, _, _ = _board_crop(page_idx, bbox_idx)
    h, w = crop.shape[:2]
    stone_dets = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    stone_dets = filter_to_grid_bbox(stone_dets, main_grid_bbox(crop))
    res = decide_edges(crop, stone_dets)

    return BoardTJunctionEdges(
        page_idx=page_idx, bbox_idx=bbox_idx,
        crop_width=w, crop_height=h,
        segments=[],
        junctions=[
            JunctionOut(
                x=j.x, y=j.y, kind=j.kind, arms=j.arms, outward=list(j.outward),
            ) for j in res.junctions
        ],
        sides={
            s: SideTallyOut(t=t.t, l=t.l, total=t.total)
            for s, t in res.sides.items()
        },
        edges=res.edges,
        stone_edges=[
            StoneEdgeClassOut(
                x=se.x, y=se.y, r=se.r, color=se.color,
                sides={d: bool(v) for d, v in se.sides.items()},
            ) for se in res.stone_edges
        ],
    )


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
    _page_bboxes_cached.cache_clear()

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
        # Emitted regardless of whether the page had any boards, so the
        # frontend can show determinate progress through the detection
        # phase even on pages where YOLO returned nothing.
        yield _json.dumps({
            "event": "page_detected", "page": page_idx + 1,
        }) + "\n"

    yield _json.dumps({
        "event": "done",
        "source": source_name,
        "total_saved": total_saved,
        "skipped": skipped,
    }) + "\n"


def _run_job(user_id: str, job_id: str) -> None:
    """Consume `_iter_ingest_events` and persist progress to job state.

    Called from a thread so the FastAPI event loop stays responsive
    while pdfium / YOLO inference runs.
    """
    pdf_path = ingest_jobs.source_pdf_path(user_id, job_id)
    if not pdf_path.exists():
        ingest_jobs.mark_error(user_id, job_id, "staged PDF missing")
        return
    state = ingest_jobs.load_state(user_id, job_id)
    if state is None:
        log.warning("ingest job %s/%s has no state; aborting", user_id, job_id)
        return
    try:
        content = pdf_path.read_bytes()
        for line in _iter_ingest_events(content, state["source"], user_id):
            try:
                event = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            ingest_jobs.apply_event(state, event)
            ingest_jobs.save_state(user_id, state)
        # Successful completion: clean up the staged PDF so we don't waste
        # space on Cloud Run / GCS. State file stays so the user sees the
        # "done" card until they dismiss it.
        ingest_jobs.cleanup_source_pdf(user_id, job_id)
    except ingest_jobs.JobDismissed:
        log.info("ingest job %s/%s dismissed mid-run; stopping", user_id, job_id)
    except Exception as e:
        log.exception("ingest job %s/%s failed", user_id, job_id)
        ingest_jobs.mark_error(user_id, job_id, str(e))


def _spawn_job(user_id: str, job_id: str) -> None:
    """Run the ingest job on a daemon thread so it works from both
    `async def` endpoints (where there's a loop) and sync ones (where
    `asyncio.get_running_loop()` would raise — those run in a worker
    thread without a loop)."""
    threading.Thread(
        target=_run_job, args=(user_id, job_id), daemon=True,
    ).start()


def _job_to_schema(state: dict) -> IngestJobOut:
    return IngestJobOut(
        job_id=state["job_id"],
        source=state.get("source", ""),
        phase=state.get("phase", "rendering"),
        started_at=state.get("started_at", ""),
        updated_at=state.get("updated_at", ""),
        total_pages=state.get("total_pages"),
        pages_rendered=state.get("pages_rendered", 0),
        pages_detected=state.get("pages_detected", 0),
        total_saved=state.get("total_saved", 0),
        skipped=state.get("skipped", 0),
        error=state.get("error"),
        stalled=bool(state.get("stalled", False)),
    )


@router.post("/ingest", response_model=IngestJobStartResponse)
async def pdf_ingest_endpoint(
    file: UploadFile = File(...), user_id: str = UserId,
) -> IngestJobStartResponse:
    """Multipart ingest (local-only). Stages the PDF and kicks off a
    background job; the client polls `/api/pdf/jobs` for progress."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    source_name = file.filename or "uploaded.pdf"
    job_id = ingest_jobs.new_job_id()
    ingest_jobs.create_job(user_id, job_id, source_name)
    ingest_jobs.source_pdf_path(user_id, job_id).write_bytes(content)
    _spawn_job(user_id, job_id)
    return IngestJobStartResponse(job_id=job_id)


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


@router.post("/ingest-from-upload", response_model=IngestJobStartResponse)
async def pdf_ingest_from_upload_endpoint(
    request: IngestFromUploadRequest,
    user_id: str = UserId,
) -> IngestJobStartResponse:
    """Ingest a PDF previously uploaded via signed URL. Stages the PDF in
    the job directory (so we can restart if the runner dies) and kicks
    off a background job. The client polls `/api/pdf/jobs`."""
    upload_path = uploads_dir(user_id) / f"{request.upload_id}.pdf"
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="upload not found")

    job_id = ingest_jobs.new_job_id()
    ingest_jobs.create_job(user_id, job_id, request.filename)
    staged = ingest_jobs.source_pdf_path(user_id, job_id)
    staged.write_bytes(upload_path.read_bytes())
    try:
        upload_path.unlink()
    except FileNotFoundError:
        pass
    _spawn_job(user_id, job_id)
    return IngestJobStartResponse(job_id=job_id)


@router.get("/jobs", response_model=IngestJobsResponse)
def pdf_jobs_endpoint(user_id: str = UserId) -> IngestJobsResponse:
    """All ingest jobs for the user, newest first."""
    return IngestJobsResponse(
        jobs=[_job_to_schema(s) for s in ingest_jobs.list_jobs(user_id)],
    )


@router.post("/jobs/{job_id}/restart", response_model=IngestJobStartResponse)
def pdf_job_restart_endpoint(
    job_id: str, user_id: str = UserId,
) -> IngestJobStartResponse:
    """Re-run a stalled or errored job from the staged PDF. The existing
    `problem_exists` short-circuit makes this resume from where it died.
    Returns a fresh job_id; the old job is deleted."""
    state = ingest_jobs.load_state(user_id, job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    src_pdf = ingest_jobs.source_pdf_path(user_id, job_id)
    if not src_pdf.exists():
        raise HTTPException(
            status_code=409,
            detail="staged PDF gone; can't restart (already cleaned up?)",
        )
    new_id = ingest_jobs.new_job_id()
    ingest_jobs.create_job(user_id, new_id, state.get("source", "uploaded.pdf"))
    ingest_jobs.source_pdf_path(user_id, new_id).write_bytes(src_pdf.read_bytes())
    ingest_jobs.delete_job(user_id, job_id)
    _spawn_job(user_id, new_id)
    return IngestJobStartResponse(job_id=new_id)


@router.delete("/jobs/{job_id}")
def pdf_job_dismiss_endpoint(job_id: str, user_id: str = UserId) -> Response:
    """Remove a job's state + staged PDF entirely."""
    if not ingest_jobs.delete_job(user_id, job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return Response(status_code=204)
