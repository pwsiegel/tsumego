import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from . import __version__
from .inference import ModelNotLoaded, detect_boards_yolo
from .schemas import (
    BboxDetectResponse,
    BboxUploadResponse,
    BoardBBoxOut,
    BoardDiscretizeLocal,
    BoardListItem,
    BoardListResponse,
    DiscretizedStoneOut,
    HealthResponse,
    Collection,
    CollectionsResponse,
    DeleteCollectionResponse,
    ProblemsResponse,
    SaveTsumegoRequest,
    SaveTsumegoResponse,
    TsumegoProblem,
    TsumegoStone,
    UpdateProblemRequest,
    UpdateProblemResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Go Problem Workbook API", version=__version__)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


# --- bbox tester -----------------------------------------------------------


@app.post("/api/pdf/bbox-upload", response_model=BboxUploadResponse)
async def bbox_upload_endpoint(file: UploadFile = File(...)) -> BboxUploadResponse:
    """Upload a PDF for bbox testing. Wipes any previous PDF, renders all
    pages to disk under BBOX_TEST_DIR, returns the page count."""
    import io
    import shutil
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from .paths import BBOX_TEST_DIR

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


@app.get("/api/pdf/bbox-page/{page_idx}.png")
def bbox_page_endpoint(page_idx: int) -> Response:
    from .paths import BBOX_TEST_DIR
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    return Response(content=path.read_bytes(), media_type="image/png")


@app.get("/api/pdf/bbox-detect/{page_idx}", response_model=BboxDetectResponse)
def bbox_detect_endpoint(page_idx: int) -> BboxDetectResponse:
    import cv2
    import numpy as np
    from .paths import BBOX_TEST_DIR
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


# ---------------------------------------------------------------------------
# Per-bbox views of the uploaded PDF. Flat list across all pages, so the
# UI can show one board per screen.
# ---------------------------------------------------------------------------


BOARD_CROP_PAD = 10   # pixels of safety context around each YOLO bbox


def _page_bboxes(page_idx: int):
    """Load the page, run YOLO, return (img, bboxes). Results are cached
    per (path, mtime) so flipping between boards doesn't re-run inference."""
    from .paths import BBOX_TEST_DIR
    import cv2
    import numpy as np
    path = BBOX_TEST_DIR / f"page_{page_idx:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"page {page_idx} not found")
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="could not decode page image")
    try:
        bboxes = _page_bboxes_cached(str(path), path.stat().st_mtime_ns)
    except ModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return img, bboxes


@lru_cache(maxsize=256)
def _page_bboxes_cached(path_str: str, _mtime_ns: int):
    import cv2
    import numpy as np
    arr = np.frombuffer(Path(path_str).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return detect_boards_yolo(img)


def _board_crop(page_idx: int, bbox_idx: int):
    """Return (crop_bgr, (x0, y0, x1, y1)_in_page) for one detected board.
    The crop is padded by BOARD_CROP_PAD px on each side so edge/pitch
    detection has room to find outer lines that YOLO may have clipped."""
    img, bboxes = _page_bboxes(page_idx)
    if bbox_idx < 0 or bbox_idx >= len(bboxes):
        raise HTTPException(
            status_code=404,
            detail=f"bbox {bbox_idx} not found on page {page_idx}",
        )
    b = bboxes[bbox_idx]
    h, w = img.shape[:2]
    P = BOARD_CROP_PAD
    x0 = max(0, b.x0 - P); y0 = max(0, b.y0 - P)
    x1 = min(w, b.x1 + P); y1 = min(h, b.y1 + P)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1), b


@app.get("/api/pdf/boards", response_model=BoardListResponse)
def boards_list_endpoint() -> BoardListResponse:
    """Flat list of every detected board across every page in the currently
    uploaded PDF, in (page, bbox-within-page) order. Drives the UI's
    one-bbox-per-screen navigation."""
    from .paths import BBOX_TEST_DIR
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


@app.get("/api/pdf/board-crop/{page_idx}/{bbox_idx}.png")
def board_crop_endpoint(page_idx: int, bbox_idx: int) -> Response:
    import cv2
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


def _discretize_board(
    page_idx: int, bbox_idx: int, peak_thresh: float = 0.3,
) -> BoardDiscretizeLocal:
    """Run the full 2b pipeline on one bbox. Shared by the HTTP endpoint
    and the batch-ingest flow."""
    from .discretize import discretize
    from .edge_inference import detect_edges
    from .pitch import measure_grid
    from .stone_inference import StoneModelNotLoaded, detect_stones_cnn
    from .stone_inference import model_available as stone_model_available
    if not stone_model_available():
        raise HTTPException(status_code=503, detail="stone detector model not trained")
    crop, _, _ = _board_crop(page_idx, bbox_idx)
    h, w = crop.shape[:2]
    try:
        stones = detect_stones_cnn(crop, peak_thresh=peak_thresh)
    except StoneModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    try:
        edges = detect_edges(crop)
    except Exception:
        edges = {"left": False, "right": False, "top": False, "bottom": False}
    grid = measure_grid(crop, edges)
    pitch = grid["pitch"]
    # Drop detections outside the detected board frame — YOLO likes to
    # fire on circular glyphs in caption text ("problem 36", etc.). Keep
    # a half-pitch margin so stones sitting right on the frame aren't
    # lost.
    # Per-axis pitch selection, in order of reliability:
    # 1. Both opposing frames detected → (R-L)/18 or (B-T)/18 (two-anchor,
    #    very accurate, captures scan anisotropy).
    # 2. One frame on that axis → that side's own per-side pitch (local to
    #    the anchor, better than a global median polluted by stones on
    #    other sides).
    # 3. Neither → global median (or None → stone-based fallback).
    if grid["left"] is not None and grid["right"] is not None:
        pitch_x = (grid["right"] - grid["left"]) / 18
    elif grid["left"] is not None:
        pitch_x = grid["left_pitch"] or pitch
    elif grid["right"] is not None:
        pitch_x = grid["right_pitch"] or pitch
    else:
        pitch_x = pitch
    if grid["top"] is not None and grid["bottom"] is not None:
        pitch_y = (grid["bottom"] - grid["top"]) / 18
    elif grid["top"] is not None:
        pitch_y = grid["top_pitch"] or pitch
    elif grid["bottom"] is not None:
        pitch_y = grid["bottom_pitch"] or pitch
    else:
        pitch_y = pitch
    # Anchor origin to detected frame edges (top/left preferred, else
    # extrapolate from bottom/right assuming a full 19-wide board).
    ox, oy = None, None
    if pitch_y is not None:
        if grid["top"] is not None:
            oy = grid["top"]
        elif grid["bottom"] is not None:
            oy_full = grid["bottom"] - 18 * pitch_y
            oy = oy_full if oy_full >= 0 else grid["bottom"] - int(grid["bottom"] / pitch_y) * pitch_y
    if pitch_x is not None:
        if grid["left"] is not None:
            ox = grid["left"]
        elif grid["right"] is not None:
            ox_full = grid["right"] - 18 * pitch_x
            ox = ox_full if ox_full >= 0 else grid["right"] - int(grid["right"] / pitch_x) * pitch_x
    # Drop detections outside the board area, and drop detections whose
    # bbox is way smaller than a real stone — YOLO fires on:
    #   - caption glyphs ("problem 36", "O", "8", …) → out-of-bounds
    #   - hoshi (star-point dots) → tiny bboxes, inside the board
    if pitch_x is not None and pitch_y is not None and pitch_x > 0 and pitch_y > 0:
        BOARD_MAX = 18  # 19x19 grid

        def _lo(frame, origin, pitch):
            if frame is not None:
                return frame - pitch * 0.5
            if origin is not None:
                return origin - pitch * 0.25
            return -1e9

        def _hi_from_origin(opposite_frame, origin, last_near, pitch, crop_size):
            """Upper bound on the board extent along one axis.

            Prefer: opposite frame → definitive.
            Else:   extrapolate from the near-side origin (origin + 18·pitch),
                    capped at the crop edge. Works even when the frame-line
                    scanner terminates early in noisy regions, which otherwise
                    would clip legitimate detections near the far side.
            Else:   fall back to the scanner's last-confirmed grid line.
            """
            if opposite_frame is not None:
                return opposite_frame + pitch * 0.5
            if origin is not None:
                return min(crop_size, origin + BOARD_MAX * pitch + pitch * 0.5)
            if last_near is not None:
                return last_near + pitch * 0.25
            return 1e9

        top_b = _lo(grid["top"], oy, pitch_y)
        bot_b = _hi_from_origin(grid["bottom"], oy, grid["top_last"], pitch_y, h)
        left_b = _lo(grid["left"], ox, pitch_x)
        right_b = _hi_from_origin(grid["right"], ox, grid["left_last"], pitch_x, w)
        stones = [
            s for s in stones
            if top_b <= s["y"] <= bot_b and left_b <= s["x"] <= right_b
        ]

        # Drop tiny detections at hoshi intersections (3/9/15, 3/9/15).
        # Hoshi dots are real ink at fixed positions; YOLO reads them
        # as small black stones. Only filter at those positions so we
        # don't accidentally drop a real stone on a board where it just
        # happens to be sitting on a hoshi.
        HOSHI = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                 (15, 3), (15, 9), (15, 15)}
        if ox is not None and oy is not None:
            vc = max(1, min(19, int((w - 1 - ox) / pitch_x) + 1))
            vr = max(1, min(19, int((h - 1 - oy) / pitch_y) + 1))
            # Window placement mirrors discretize._place_window
            cmin = 0 if edges.get("left") else (19 - vc if edges.get("right") else max(0, (19 - vc) // 2))
            rmin = 0 if edges.get("top") else (19 - vr if edges.get("bottom") else max(0, (19 - vr) // 2))
            min_pitch = min(pitch_x, pitch_y)
            # Hoshi detections on real scans come back with r ≈ 0.15-0.25·pitch
            # (YOLO pads a small bbox around the dark dot). Real stones sit at
            # r ≈ 0.4·pitch. A 0.28 threshold cleanly separates them.
            hoshi_r = 0.28 * min_pitch
            def _not_hoshi(s: dict) -> bool:
                cl = max(0, min(vc - 1, int(round((s["x"] - ox) / pitch_x))))
                rl = max(0, min(vr - 1, int(round((s["y"] - oy) / pitch_y))))
                if (cmin + cl, rmin + rl) in HOSHI and s.get("r", 0) < hoshi_r:
                    return False
                return True
            stones = [s for s in stones if _not_hoshi(s)]
    d = discretize(
        stones, w, h, edges=edges,
        cell_size_override=pitch,
        pitch_x_override=pitch_x, pitch_y_override=pitch_y,
        origin_x_override=ox, origin_y_override=oy,
    )
    return BoardDiscretizeLocal(
        page_idx=page_idx, bbox_idx=bbox_idx,
        crop_width=w, crop_height=h,
        cell_size=d.cell_size,
        origin_x=d.origin_x, origin_y=d.origin_y,
        visible_cols=d.visible_cols, visible_rows=d.visible_rows,
        col_min=d.col_min, row_min=d.row_min,
        edges=edges,
        stones=[DiscretizedStoneOut(
            x=s.x, y=s.y, color=s.color, conf=s.conf,
            col_local=s.col_local, row_local=s.row_local,
            col=s.col, row=s.row,
        ) for s in d.stones],
    )


@app.get("/api/pdf/board-discretize/{page_idx}/{bbox_idx}",
         response_model=BoardDiscretizeLocal)
def board_discretize_endpoint(
    page_idx: int, bbox_idx: int,
    peak_thresh: float = 0.3,
) -> BoardDiscretizeLocal:
    """End-to-end 2b pipeline for a single bbox."""
    return _discretize_board(page_idx, bbox_idx, peak_thresh)


@app.post("/api/tsumego/save", response_model=SaveTsumegoResponse)
def tsumego_save_endpoint(req: SaveTsumegoRequest) -> SaveTsumegoResponse:
    """Persist an accepted problem as SGF + metadata JSON + crop PNG."""
    import cv2
    from .tsumego import save_problem

    # Grab the crop for this page/bbox from the currently-uploaded PDF
    # and encode as PNG. The source PDF is still on disk in BBOX_TEST_DIR
    # for the duration of the upload-review flow, so this is always
    # available at the moment Accept is clicked.
    crop_png: bytes | None = None
    try:
        crop, _, _ = _board_crop(req.page_idx, req.bbox_idx)
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            crop_png = buf.tobytes()
    except HTTPException:
        # If the crop's gone (e.g. PDF was swapped), still save the SGF.
        pass

    stones = [s.model_dump() for s in req.stones]
    path = save_problem(
        source=req.source,
        uploaded_at=req.uploaded_at,
        source_board_idx=req.source_board_idx,
        stones=stones,
        black_to_play=req.black_to_play,
        crop_png=crop_png,
        status=req.status,
    )
    return SaveTsumegoResponse(id=path.stem)


@app.get("/api/tsumego/collections", response_model=CollectionsResponse)
def tsumego_collections_endpoint() -> CollectionsResponse:
    from .tsumego import list_collections
    items = list_collections()
    return CollectionsResponse(
        collections=[Collection(**c) for c in items]
    )


@app.delete("/api/tsumego/collections/{source:path}",
            response_model=DeleteCollectionResponse)
def tsumego_collection_delete_endpoint(source: str) -> DeleteCollectionResponse:
    """Remove every saved problem whose source matches `source`.
    Path uses :path so names with dots (the usual case) pass through."""
    from .tsumego import delete_collection
    removed = delete_collection(source)
    return DeleteCollectionResponse(source=source, removed=removed)


@app.post("/api/pdf/ingest")
async def pdf_ingest_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a PDF, render pages, run the full detect→discretize pipeline
    on every board, save each as an unreviewed problem. Streams NDJSON
    progress events so the UI can show per-board progress."""
    import io
    import json as _json
    import shutil
    import time
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from .paths import BBOX_TEST_DIR
    from .tsumego import save_problem, problem_exists

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    source_name = file.filename or "uploaded.pdf"
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Wipe the bbox_test staging dir first — the detect/discretize
    # endpoints key off of these per-page PNGs.
    if BBOX_TEST_DIR.exists():
        shutil.rmtree(BBOX_TEST_DIR)
    BBOX_TEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid PDF: {e}") from e
    total_pages = len(pdf)

    def iter_events():
        yield _json.dumps({
            "event": "start", "source": source_name,
            "uploaded_at": uploaded_at, "total_pages": total_pages,
        }) + "\n"

        # First pass: render every page so board-detect caches work.
        for i, page in enumerate(pdf):
            pil_img = page.render(scale=2.0).to_pil()
            img_bgr = np.array(pil_img)[..., ::-1].copy()
            ok, buf = cv2.imencode(".png", img_bgr)
            if ok:
                (BBOX_TEST_DIR / f"page_{i:04d}.png").write_bytes(buf.tobytes())
            yield _json.dumps({
                "event": "page_rendered", "page": i + 1, "total_pages": total_pages,
            }) + "\n"

        # Second pass: detect boards on each page, then discretize and
        # save. We emit the flat source_board_idx as each board is saved.
        total_saved = 0
        skipped = 0
        source_board_idx = 0
        for page_idx in range(total_pages):
            _, bboxes = _page_bboxes(page_idx)
            for bbox_idx in range(len(bboxes)):
                # Skip if we've already saved a problem here (re-ingest).
                if problem_exists(source_name, source_board_idx):
                    skipped += 1
                    source_board_idx += 1
                    continue
                try:
                    d = _discretize_board(page_idx, bbox_idx)
                    crop, _, _ = _board_crop(page_idx, bbox_idx)
                    ok, buf = cv2.imencode(".png", crop)
                    crop_png = buf.tobytes() if ok else None
                    save_problem(
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

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


@app.get("/api/tsumego/collections/{source:path}/problems",
         response_model=ProblemsResponse)
def tsumego_collection_problems_endpoint(source: str) -> ProblemsResponse:
    from .tsumego import list_problems
    items = list_problems(source)
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


@app.get("/api/tsumego/{problem_id}/image.png")
def tsumego_image_endpoint(problem_id: str) -> Response:
    from .paths import TSUMEGO_DIR
    from .tsumego import load_problem
    meta = load_problem(problem_id)
    if meta is None or not meta.get("image"):
        raise HTTPException(status_code=404, detail="image not found")
    path = TSUMEGO_DIR / meta["image"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return Response(content=path.read_bytes(), media_type="image/png")


@app.get("/api/tsumego/{problem_id}", response_model=TsumegoProblem)
def tsumego_get_endpoint(problem_id: str) -> TsumegoProblem:
    from .tsumego import load_problem
    p = load_problem(problem_id)
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


@app.post("/api/val/{dataset}/problems/{stem}/stones")
def val_update_stones_endpoint(dataset: str, stem: str, req: dict) -> dict:
    """Overwrite the ground-truth stones for a problem in a val dataset.

    Rewrites metadata/{stem}.json (authoritative), regenerates the SGF,
    and updates the GT + matches_gt flags in the pre-computed
    comparison.json so the UI reflects the edit on the next fetch.
    Also appends an entry to gt_edits.json so the user has an audit
    trail of annotation changes for later propagation to other stores.
    """
    import json as _json
    import time
    from .paths import DATA_DIR
    from .tsumego import stones_to_sgf

    stones = req.get("stones")
    if not isinstance(stones, list):
        raise HTTPException(status_code=400, detail="expected stones: list")
    norm = [
        {"col": int(s["col"]), "row": int(s["row"]), "color": str(s["color"])}
        for s in stones
    ]

    val_dir = DATA_DIR / "val" / dataset
    meta_path = val_dir / "metadata" / f"{stem}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"{stem} not found in val/{dataset}")
    meta = _json.loads(meta_path.read_text())
    before = meta.get("stones", [])
    meta["stones"] = norm

    image_filename = meta.get("image") or meta.get("sgf", "").replace(".sgf", ".png")
    sgf_text = stones_to_sgf(
        norm, bool(meta.get("black_to_play", True)),
        image_ref=f"./{image_filename}" if image_filename else None,
    )

    meta_path.write_text(_json.dumps(meta, indent=2))
    sgf_path = val_dir / "sgf" / f"{stem}.sgf"
    sgf_path.write_text(sgf_text)

    # Sync comparison.json if present so the UI's cached source of truth
    # also picks up the edit.
    comp_path = val_dir / "comparison.json"
    if comp_path.exists():
        comp = _json.loads(comp_path.read_text())
        for p in comp.get("problems", []):
            if p["stem"] == stem:
                p["gt"] = norm
                gt_set = {(s["col"], s["row"], s["color"]) for s in norm}
                old_set = {(s["col"], s["row"], s["color"]) for s in p["old"]}
                new_set = {(s["col"], s["row"], s["color"]) for s in p["new"]}
                p["old_matches_gt"] = old_set == gt_set
                p["new_matches_gt"] = new_set == gt_set
                break
        comp_path.write_text(_json.dumps(comp, indent=2))

    # Audit-trail log: gt_edits.json is a chronological list of every
    # ground-truth edit made through this UI. Each entry records before/
    # after + a diff summary so edits can be propagated to the live
    # tsumego collection (or elsewhere) later without opening every file.
    before_set = {(s["col"], s["row"], s["color"]) for s in before}
    after_set = {(s["col"], s["row"], s["color"]) for s in norm}
    before_pos = {(s["col"], s["row"]): s["color"] for s in before}
    after_pos = {(s["col"], s["row"]): s["color"] for s in norm}
    color_flips = [
        {"col": c, "row": r, "from": before_pos[(c, r)], "to": after_pos[(c, r)]}
        for (c, r) in before_pos.keys() & after_pos.keys()
        if before_pos[(c, r)] != after_pos[(c, r)]
    ]
    flip_positions = {(f["col"], f["row"]) for f in color_flips}
    added = sorted(
        {(c, r, col) for (c, r, col) in after_set - before_set
         if (c, r) not in flip_positions}
    )
    removed = sorted(
        {(c, r, col) for (c, r, col) in before_set - after_set
         if (c, r) not in flip_positions}
    )
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stem": stem,
        "source": meta.get("source"),
        "source_board_idx": meta.get("source_board_idx"),
        "original_tsumego_id": meta.get("id"),
        "before": before,
        "after": norm,
        "added": [{"col": c, "row": r, "color": col} for (c, r, col) in added],
        "removed": [{"col": c, "row": r, "color": col} for (c, r, col) in removed],
        "color_flips": color_flips,
    }
    edits_path = val_dir / "gt_edits.json"
    edits: list[dict] = []
    if edits_path.exists():
        try:
            edits = _json.loads(edits_path.read_text())
        except _json.JSONDecodeError:
            edits = []
    edits.append(entry)
    edits_path.write_text(_json.dumps(edits, indent=2))

    return {"ok": True, "stem": stem, "stones": norm}


@app.get("/api/val/{dataset}/gt-edits")
def val_gt_edits_endpoint(dataset: str) -> Response:
    """Chronological log of ground-truth edits made through the compare UI."""
    import json as _json
    from .paths import DATA_DIR
    path = DATA_DIR / "val" / dataset / "gt_edits.json"
    if not path.exists():
        return Response(content=_json.dumps([]), media_type="application/json")
    return Response(content=path.read_bytes(), media_type="application/json")


@app.get("/api/val/{dataset}/comparison")
def val_comparison_endpoint(dataset: str) -> Response:
    """Serve the comparison JSON emitted by generate_comparison.py."""
    from .paths import DATA_DIR
    path = DATA_DIR / "val" / dataset / "comparison.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no comparison.json in val/{dataset}")
    return Response(content=path.read_bytes(), media_type="application/json")


@app.get("/api/val/{dataset}/images/{stem}.png")
def val_image_endpoint(dataset: str, stem: str) -> Response:
    from .paths import DATA_DIR
    path = DATA_DIR / "val" / dataset / "images" / f"{stem}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"image not found: {stem}")
    return Response(content=path.read_bytes(), media_type="image/png")


@app.delete("/api/tsumego/{problem_id}")
def tsumego_delete_endpoint(problem_id: str) -> dict:
    from .tsumego import delete_problem
    if not delete_problem(problem_id):
        raise HTTPException(status_code=404, detail=problem_id)
    return {"id": problem_id, "deleted": True}


@app.post("/api/tsumego/{problem_id}/status", response_model=UpdateProblemResponse)
def tsumego_update_endpoint(
    problem_id: str, req: UpdateProblemRequest,
) -> UpdateProblemResponse:
    from .tsumego import update_problem
    stones = [s.model_dump() for s in req.stones] if req.stones is not None else None
    try:
        meta = update_problem(
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

