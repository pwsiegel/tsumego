"""Interactive PDF ingest: render+detect every page, let the user skip
bad bboxes and add missing ones, then ingest the surviving set fresh.

This is the only path the upload UI uses. (The headless `pdf_ingest`
path still exists for the CLI / legacy `/api/pdf/ingest*` endpoints.)

A session lives under `data/patch_sessions/{user_id}/{session_id}/`:

    source.pdf            staged PDF (deleted on dismiss)
    state.json            phase + per-page detected bboxes
    pages/page_NNNN.png   rendered page images (kept until session ends so
                          apply can crop new bboxes without re-rendering)

State schema:

    {
      session_id, user_id, source,
      phase: rendering | detecting | ready | applying | done | error,
      started_at, updated_at,
      total_pages, pages_rendered, pages_detected,
      pages: {
        "<page_idx>": {
          "image_w": int, "image_h": int,
          "bboxes": [{"bbox_idx", "x0", "y0", "x1", "y1"}]
        }
      },
      apply: {                  # populated once the user POSTs apply
        "total": int,           # number of bboxes to ingest
        "ingested": int,        # successfully saved so far
        "failed": int,          # discretize errors so far
      },
      error: str | null
    }

Apply consumes the session state plus an edits payload (per-page skips
+ adds), runs the existing crop→discretize→SGF pipeline on the kept +
added bboxes, and assigns `source_board_idx` contiguously in
(page_idx, bbox_idx) order across the whole PDF.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .paths import DATA_DIR

log = logging.getLogger(__name__)

PATCH_SESSIONS_ROOT = DATA_DIR / "patch_sessions"
TERMINAL_PHASES = {"done", "error"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def session_dir(user_id: str, session_id: str) -> Path:
    return PATCH_SESSIONS_ROOT / user_id / session_id


def pages_dir(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "pages"


def page_image_path(user_id: str, session_id: str, page_idx: int) -> Path:
    return pages_dir(user_id, session_id) / f"page_{page_idx:04d}.png"


def source_pdf_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "source.pdf"


def state_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "state.json"


def new_session_id() -> str:
    return uuid.uuid4().hex


def create_session(user_id: str, session_id: str, source: str) -> dict[str, Any]:
    sdir = session_dir(user_id, session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    pages_dir(user_id, session_id).mkdir(parents=True, exist_ok=True)
    state = {
        "session_id": session_id,
        "user_id": user_id,
        "source": source,
        "phase": "rendering",
        "started_at": _now(),
        "updated_at": _now(),
        "total_pages": None,
        "pages_rendered": 0,
        "pages_detected": 0,
        "pages": {},
        "apply": None,
        "error": None,
    }
    save_state(state)
    return state


def load_state(user_id: str, session_id: str) -> dict[str, Any] | None:
    p = state_path(user_id, session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("patch_sessions: bad state %s: %s", p, e)
        return None


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    p = state_path(state["user_id"], state["session_id"])
    if not p.parent.exists():
        # User dismissed the session; don't resurrect it.
        return
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    root = PATCH_SESSIONS_ROOT / user_id
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        s = load_state(user_id, child.name)
        if s is not None:
            out.append(s)
    out.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return out


def mark_error(user_id: str, session_id: str, message: str) -> None:
    s = load_state(user_id, session_id)
    if s is None:
        return
    s["phase"] = "error"
    s["error"] = message
    save_state(s)


def delete_session(user_id: str, session_id: str) -> bool:
    sdir = session_dir(user_id, session_id)
    if not sdir.exists():
        return False
    pdir = pages_dir(user_id, session_id)
    if pdir.exists():
        for f in pdir.iterdir():
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        try:
            pdir.rmdir()
        except OSError:
            pass
    for f in sdir.iterdir():
        try:
            f.unlink()
        except (FileNotFoundError, IsADirectoryError):
            pass
    try:
        sdir.rmdir()
    except OSError:
        pass
    return True


# --- run: render every page + detect bboxes ---


def run_session(user_id: str, session_id: str) -> None:
    """Render+detect every page, save bboxes to state.

    Reuses pdf_ingest's parallel page processor but only the render+
    detect prefix — no save. Each page's PNG is written to pages_dir
    so apply() can crop new bboxes from cached pixels.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .pdf_ingest import _worker_init, default_workers

    state = load_state(user_id, session_id)
    if state is None:
        log.warning("patch session %s/%s missing state", user_id, session_id)
        return

    pdf_path = source_pdf_path(user_id, session_id)
    if not pdf_path.exists():
        mark_error(user_id, session_id, "staged PDF missing")
        return

    try:
        import pypdfium2 as pdfium
        n_pages = len(pdfium.PdfDocument(str(pdf_path)))
    except Exception as e:
        mark_error(user_id, session_id, f"could not open PDF: {e}")
        return

    state["total_pages"] = n_pages
    state["phase"] = "rendering"
    save_state(state)

    workers = default_workers()
    sdir_str = str(pages_dir(user_id, session_id))

    completed = 0
    page_results: dict[int, dict] = {}
    try:
        if workers <= 1:
            for i in range(n_pages):
                page_results[i] = _render_and_detect_page(str(pdf_path), i, sdir_str)
                completed += 1
                state["pages_rendered"] = completed
                state["pages_detected"] = completed
                save_state(state)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx, initializer=_worker_init,
            ) as ex:
                futs = [
                    ex.submit(_render_and_detect_page, str(pdf_path), i, sdir_str)
                    for i in range(n_pages)
                ]
                for fut in as_completed(futs):
                    res = fut.result()
                    page_results[res["page_idx"]] = res
                    completed += 1
                    state["pages_rendered"] = completed
                    state["pages_detected"] = completed
                    if completed % 5 == 0 or completed == n_pages:
                        save_state(state)
    except Exception as e:
        log.exception("patch session %s/%s render+detect failed", user_id, session_id)
        mark_error(user_id, session_id, str(e))
        return

    pages_state: dict[str, dict] = {}
    for page_idx in range(n_pages):
        res = page_results.get(page_idx, {"bboxes": [], "image_w": 0, "image_h": 0})
        pages_state[str(page_idx)] = {
            "image_w": res["image_w"],
            "image_h": res["image_h"],
            "bboxes": res["bboxes"],
        }

    state["pages"] = pages_state
    state["phase"] = "ready"
    save_state(state)


def _render_and_detect_page(pdf_path: str, page_idx: int, pages_dir_str: str) -> dict:
    """Worker: render one page to PNG and return bboxes."""
    import cv2
    import numpy as np
    import pypdfium2 as pdfium

    from .ml.board_detect.detect import detect_boards_yolo

    pdf = pdfium.PdfDocument(pdf_path)
    pil = pdf[page_idx].render(scale=2.0).to_pil()
    img_bgr = np.array(pil)[..., ::-1].copy()
    h, w = img_bgr.shape[:2]
    out_path = Path(pages_dir_str) / f"page_{page_idx:04d}.png"
    ok, buf = cv2.imencode(".png", img_bgr)
    if ok:
        out_path.write_bytes(buf.tobytes())
    bboxes = detect_boards_yolo(img_bgr)
    return {
        "page_idx": page_idx,
        "image_w": w,
        "image_h": h,
        "bboxes": [
            {"bbox_idx": i, "x0": int(b.x0), "y0": int(b.y0),
             "x1": int(b.x1), "y1": int(b.y1)}
            for i, b in enumerate(bboxes)
        ],
    }


# --- apply: ingest the kept + added bboxes ---


def apply_session(
    user_id: str, session_id: str, edits: dict[str, Any],
) -> dict[str, Any]:
    """Ingest every detected bbox the user did not skip, plus every
    user-added bbox.

    `edits` shape:
        {
          "skip": [{"page_idx": int, "bbox_idx": int}, ...],
          "adds": [
            {"page_idx": int, "x0": int, "y0": int, "x1": int, "y1": int}
          ],
        }

    `source_board_idx` is assigned contiguously in (page_idx, bbox_idx)
    order, with adds appended after the detected bboxes on each page.

    Updates `state["apply"] = {total, ingested, failed}` periodically so
    the home page progress bar can poll. On completion, deletes the
    staged PDF and rendered page PNGs (state.json stays so the "done"
    card persists until the user dismisses it).
    """
    import cv2
    import numpy as np

    from .ml.pipeline import BOARD_CROP_PAD, discretize_crop
    from .tsumego import save_problem

    state = load_state(user_id, session_id)
    if state is None:
        raise FileNotFoundError(f"session {session_id} not found")

    source = state["source"]
    uploaded_at = state.get("started_at") or _now()

    skip_set: set[tuple[int, int]] = {
        (int(s["page_idx"]), int(s["bbox_idx"])) for s in edits.get("skip", [])
    }
    adds_by_page: dict[int, list[dict]] = {}
    for a in edits.get("adds", []):
        adds_by_page.setdefault(int(a["page_idx"]), []).append(a)
    for page_adds in adds_by_page.values():
        # Sort top-to-bottom, left-to-right so bbox_idx assignment matches
        # natural reading order.
        page_adds.sort(key=lambda a: (a["y0"], a["x0"]))

    pages_state = state.get("pages") or {}
    page_indices = sorted(int(k) for k in pages_state.keys())

    # Build the full ingest list in (page_idx, bbox_idx) order.
    ingest_plan: list[dict] = []
    for page_idx in page_indices:
        page = pages_state[str(page_idx)]
        kept = [
            b for b in page.get("bboxes", [])
            if (page_idx, int(b["bbox_idx"])) not in skip_set
        ]
        next_idx = (
            max((int(b["bbox_idx"]) for b in page.get("bboxes", [])), default=-1) + 1
        )
        for b in kept:
            ingest_plan.append({
                "page_idx": page_idx,
                "bbox_idx": int(b["bbox_idx"]),
                "x0": int(b["x0"]), "y0": int(b["y0"]),
                "x1": int(b["x1"]), "y1": int(b["y1"]),
            })
        for a in adds_by_page.get(page_idx, []):
            ingest_plan.append({
                "page_idx": page_idx,
                "bbox_idx": next_idx,
                "x0": int(a["x0"]), "y0": int(a["y0"]),
                "x1": int(a["x1"]), "y1": int(a["y1"]),
            })
            next_idx += 1

    state["phase"] = "applying"
    state["apply"] = {"total": len(ingest_plan), "ingested": 0, "failed": 0}
    save_state(state)

    ingested = 0
    failed = 0
    page_imgs: dict[int, np.ndarray] = {}

    for sbi, item in enumerate(ingest_plan):
        page_idx = item["page_idx"]
        if page_idx not in page_imgs:
            p = page_image_path(user_id, session_id, page_idx)
            if not p.exists():
                log.warning("patch apply: page %d image missing", page_idx)
                failed += 1
                _bump_apply_progress(user_id, session_id, ingested, failed)
                continue
            img = cv2.imdecode(
                np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR,
            )
            if img is None:
                log.warning("patch apply: could not decode page %d", page_idx)
                failed += 1
                _bump_apply_progress(user_id, session_id, ingested, failed)
                continue
            page_imgs[page_idx] = img
        img = page_imgs[page_idx]
        h, w = img.shape[:2]

        x0 = max(0, item["x0"] - BOARD_CROP_PAD)
        y0 = max(0, item["y0"] - BOARD_CROP_PAD)
        x1 = min(w, item["x1"] + BOARD_CROP_PAD)
        y1 = min(h, item["y1"] + BOARD_CROP_PAD)
        crop = img[y0:y1, x0:x1]
        try:
            d, _ = discretize_crop(crop)
            stones = [
                {"col": int(s.col), "row": int(s.row), "color": str(s.color)}
                for s in d.stones
            ]
            ok, buf = cv2.imencode(".png", crop)
            save_problem(
                user_id=user_id,
                source=source,
                uploaded_at=uploaded_at,
                source_board_idx=sbi,
                stones=stones,
                black_to_play=True,
                crop_png=buf.tobytes() if ok else None,
                status="unreviewed",
                page_idx=page_idx,
                bbox_idx=item["bbox_idx"],
            )
            ingested += 1
        except Exception as e:
            log.warning(
                "patch apply: discretize failed page %d bbox %d: %s",
                page_idx, item["bbox_idx"], e,
            )
            failed += 1
        # Save every 5 boards (and always on the last one) so the home
        # page poller sees fresh progress without thrashing the disk.
        if (sbi + 1) % 5 == 0 or (sbi + 1) == len(ingest_plan):
            _bump_apply_progress(user_id, session_id, ingested, failed)

    state = load_state(user_id, session_id) or state
    state["phase"] = "done"
    state["apply"] = {
        "total": len(ingest_plan), "ingested": ingested, "failed": failed,
    }
    save_state(state)
    _cleanup_artifacts(user_id, session_id)
    return {
        "ingested": ingested,
        "skipped": len(skip_set),
        "failed": failed,
    }


def _bump_apply_progress(
    user_id: str, session_id: str, ingested: int, failed: int,
) -> None:
    s = load_state(user_id, session_id)
    if s is None or s.get("apply") is None:
        return
    s["apply"]["ingested"] = ingested
    s["apply"]["failed"] = failed
    save_state(s)


def _cleanup_artifacts(user_id: str, session_id: str) -> None:
    """Drop the staged PDF + rendered pages once apply is done.

    Keeps state.json so the home-page card persists until the user
    dismisses it (mirroring how `ingest_jobs.cleanup_source_pdf` works)."""
    pdir = pages_dir(user_id, session_id)
    if pdir.exists():
        for f in pdir.iterdir():
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        try:
            pdir.rmdir()
        except OSError:
            pass
    pdf = source_pdf_path(user_id, session_id)
    try:
        pdf.unlink()
    except FileNotFoundError:
        pass
