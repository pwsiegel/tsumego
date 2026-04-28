"""Parallel PDF → board-detection → save pipeline.

Used by both the API ingest job runner (`api.pdf.routes._run_job`) and the
local CLI (`cli.ingest_pdf`). Each page is rendered + detected in a
worker process; the main process collects results in page order and
calls `save_problem` serially so `source_board_idx` is assigned
deterministically.

Workers default to `cpu_count - 2`, which gives 1 on Cloud Run's 2-vCPU
runtime (effectively in-process — no pool overhead, the warm ONNX
session in the parent process is reused) and ~12 on a laptop. Each
worker pins OMP/MKL/OpenBLAS/cv2 to a single thread so N workers don't
oversubscribe the CPU.

NDJSON event stream is compatible with `ingest_jobs.apply_event`:
    start → page_rendered* → page_detected* → board_saved* → done.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

from .tsumego import problem_exists, save_problem

log = logging.getLogger(__name__)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)


def _process_page(pdf_path: str, page_idx: int) -> tuple[int, list[dict]]:
    """Render one page, detect boards, discretize each. Returns (page_idx, boards)."""
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from .ml.board_detect.detect import detect_boards_yolo
    from .ml.pipeline import BOARD_CROP_PAD, discretize_crop

    pdf = pdfium.PdfDocument(pdf_path)
    pil = pdf[page_idx].render(scale=2.0).to_pil()
    img_bgr = np.array(pil)[..., ::-1].copy()

    bboxes = detect_boards_yolo(img_bgr)
    h, w = img_bgr.shape[:2]
    pad = BOARD_CROP_PAD

    boards: list[dict] = []
    for bbox_idx, b in enumerate(bboxes):
        x0 = max(0, b.x0 - pad); y0 = max(0, b.y0 - pad)
        x1 = min(w, b.x1 + pad); y1 = min(h, b.y1 + pad)
        crop = img_bgr[y0:y1, x0:x1]
        try:
            d, _ = discretize_crop(crop)
            stones = [
                {"col": int(s.col), "row": int(s.row), "color": str(s.color)}
                for s in d.stones
            ]
            ok, buf = cv2.imencode(".png", crop)
            boards.append({
                "bbox_idx": bbox_idx,
                "stones": stones,
                "crop_png": buf.tobytes() if ok else None,
            })
        except Exception as e:
            boards.append({"bbox_idx": bbox_idx, "error": repr(e)})
    return page_idx, boards


def _count_pages(pdf_path: str) -> int:
    import pypdfium2 as pdfium
    return len(pdfium.PdfDocument(pdf_path))


def iter_ingest_events(
    pdf_path: str,
    source_name: str,
    user_id: str,
    workers: int | None = None,
) -> Iterator[str]:
    """Yield NDJSON events compatible with `ingest_jobs.apply_event`.

    With workers=1 the loop runs in-process so the warm ONNX session in
    the parent is reused; with workers>1 it fans out via a spawn-mode
    ProcessPoolExecutor.
    """
    if workers is None:
        workers = default_workers()
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_pages = _count_pages(pdf_path)

    yield json.dumps({
        "event": "start", "source": source_name,
        "uploaded_at": uploaded_at, "total_pages": total_pages,
    }) + "\n"

    results: dict[int, list[dict]] = {}
    state = {"next_to_save": 0, "source_board_idx": 0,
             "total_saved": 0, "skipped": 0}

    def drain() -> Iterator[str]:
        """Save problems for any consecutive completed pages from
        `next_to_save` onward; yield board_saved + page_detected events.
        Pages stay buffered until their predecessor has been saved, so
        `source_board_idx` matches the serial-ingest numbering."""
        while state["next_to_save"] in results:
            page_idx = state["next_to_save"]
            for board in results.pop(page_idx):
                sbi = state["source_board_idx"]
                if "error" in board:
                    log.warning("ingest: page %d bbox %d failed: %s",
                                page_idx, board.get("bbox_idx"), board["error"])
                elif problem_exists(user_id, source_name, sbi):
                    state["skipped"] += 1
                else:
                    save_problem(
                        user_id=user_id,
                        source=source_name,
                        uploaded_at=uploaded_at,
                        source_board_idx=sbi,
                        stones=board["stones"],
                        black_to_play=True,
                        crop_png=board["crop_png"],
                        status="unreviewed",
                        page_idx=page_idx,
                        bbox_idx=board["bbox_idx"],
                    )
                    state["total_saved"] += 1
                state["source_board_idx"] = sbi + 1
                yield json.dumps({
                    "event": "board_saved",
                    "source_board_idx": sbi,
                    "page_idx": page_idx,
                    "bbox_idx": board.get("bbox_idx"),
                    "total_saved": state["total_saved"],
                }) + "\n"
            yield json.dumps({
                "event": "page_detected", "page": page_idx + 1,
            }) + "\n"
            state["next_to_save"] += 1

    completed = 0
    if workers <= 1:
        for page_idx in range(total_pages):
            _, boards = _process_page(pdf_path, page_idx)
            results[page_idx] = boards
            completed += 1
            yield json.dumps({
                "event": "page_rendered", "page": completed,
                "total_pages": total_pages,
            }) + "\n"
            yield from drain()
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
        ) as ex:
            futures = [
                ex.submit(_process_page, pdf_path, i)
                for i in range(total_pages)
            ]
            for fut in as_completed(futures):
                page_idx, boards = fut.result()
                results[page_idx] = boards
                completed += 1
                yield json.dumps({
                    "event": "page_rendered", "page": completed,
                    "total_pages": total_pages,
                }) + "\n"
                yield from drain()

    yield json.dumps({
        "event": "done", "source": source_name,
        "total_saved": state["total_saved"],
        "skipped": state["skipped"],
    }) + "\n"
