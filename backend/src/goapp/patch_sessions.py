"""Bbox-edit patch sessions: re-detect a previously-ingested PDF, let the
user delete + add bboxes, then patch the existing collection in place.

A session lives under `data/patch_sessions/{user_id}/{session_id}/`:

    source.pdf            staged PDF (deleted on dismiss)
    state.json            phase + per-page bbox info + alignment
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
          "bboxes": [
            {"bbox_idx", "x0", "y0", "x1", "y1",
             "existing_problem_id": <str|null>}
          ]
        }
      },
      align_warnings: [str],   # e.g. "page 4: 2 detected vs 3 saved"
      error: str | null
    }

Identity comes from the (page_idx, bbox_idx) → existing_problem_id map
built at start time, by re-running YOLO and matching against the
existing problems on disk for `source`. YOLO is deterministic, so a
clean alignment means each detected bbox lines up with exactly one
ingested problem; mismatches are surfaced as `align_warnings` for the
user to inspect before applying.

Apply consumes the session state plus an edits payload (per-page deletes
+ adds), runs the existing crop→discretize→SGF pipeline on adds, deletes
the gone problems, and reindexes `source_board_idx` contiguously across
the whole source.
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
        "align_warnings": [],
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
    # Remove pages dir contents first, then top-level files, then dirs.
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


# --- run: render every page + detect bboxes + align to existing problems ---


def run_session(user_id: str, session_id: str) -> None:
    """Render+detect every page, save bboxes to state, align to disk.

    Reuses pdf_ingest's parallel page processor but only the render+
    detect prefix — no save. Each page's PNG is written to pages_dir
    so apply() can crop new bboxes from cached pixels.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .pdf_ingest import _worker_init, default_workers
    from .tsumego import list_problems

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

    # Build the (page_idx, bbox_idx) → existing_problem_id alignment map.
    existing = list_problems(user_id, state["source"])
    by_page: dict[int, list[dict]] = {}
    for p in existing:
        pi = p.get("page_idx")
        if pi is None:
            continue
        by_page.setdefault(pi, []).append(p)

    pages_state: dict[str, dict] = {}
    warnings: list[str] = []
    for page_idx in range(n_pages):
        res = page_results.get(page_idx, {"bboxes": [], "image_w": 0, "image_h": 0})
        existing_on_page = sorted(
            by_page.get(page_idx, []), key=lambda d: d.get("bbox_idx", 0),
        )
        # Index-based alignment: problem with bbox_idx==i ↔ detected bbox i.
        existing_by_idx = {p.get("bbox_idx"): p["id"] for p in existing_on_page}
        bb_out: list[dict] = []
        for b in res["bboxes"]:
            pid = existing_by_idx.get(b["bbox_idx"])
            bb_out.append({**b, "existing_problem_id": pid})
        if existing_on_page and len(res["bboxes"]) != len(existing_on_page):
            warnings.append(
                f"page {page_idx}: {len(res['bboxes'])} detected vs "
                f"{len(existing_on_page)} saved"
            )
        pages_state[str(page_idx)] = {
            "image_w": res["image_w"],
            "image_h": res["image_h"],
            "bboxes": bb_out,
        }

    state["pages"] = pages_state
    state["align_warnings"] = warnings
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


# --- apply: turn an edits payload into deletes + adds + reindex ---


def apply_session(
    user_id: str, session_id: str, edits: dict[str, Any],
) -> dict[str, Any]:
    """Mutate the collection per `edits` and reindex.

    `edits` shape:
        {
          "deletes": ["<problem_id>", ...],         # IDs to remove
          "adds": [
            {"page_idx": int, "x0": int, "y0": int, "x1": int, "y1": int}
          ],
        }

    Returns: {"deleted": int, "added": int, "reindexed": int}.
    """
    import cv2
    import numpy as np

    from .ml.pipeline import BOARD_CROP_PAD, discretize_crop
    from .paths import tsumego_dir
    from .tsumego import delete_problem, list_problems, save_problem

    state = load_state(user_id, session_id)
    if state is None:
        raise FileNotFoundError(f"session {session_id} not found")
    state["phase"] = "applying"
    save_state(state)

    source = state["source"]
    uploaded_at = state.get("started_at") or _now()

    deletes = list(edits.get("deletes", []))
    adds = list(edits.get("adds", []))

    deleted = 0
    for pid in deletes:
        if delete_problem(user_id, pid):
            deleted += 1

    added = 0
    # Bucket adds by page so we can compute next-available bbox_idx per page.
    adds_by_page: dict[int, list[dict]] = {}
    for a in adds:
        adds_by_page.setdefault(int(a["page_idx"]), []).append(a)

    # For each page receiving an add, figure out the next bbox_idx so it
    # sorts after any existing bboxes there. After deletions we ignore
    # the gone problems' indices — we want the new one to be > any
    # surviving index on the page.
    surviving_by_page = _problems_grouped_by_page(user_id, source)

    for page_idx, page_adds in adds_by_page.items():
        page_path = page_image_path(user_id, session_id, page_idx)
        if not page_path.exists():
            log.warning("patch apply: page %d image missing, skipping %d adds",
                        page_idx, len(page_adds))
            continue
        img = cv2.imdecode(
            np.frombuffer(page_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR,
        )
        if img is None:
            log.warning("patch apply: could not decode page %d", page_idx)
            continue
        h, w = img.shape[:2]

        existing_idxs = [
            p.get("bbox_idx", -1) for p in surviving_by_page.get(page_idx, [])
        ]
        next_idx = (max(existing_idxs) + 1) if existing_idxs else 0
        # Sort adds top-to-bottom, left-to-right so bbox_idx assignment
        # matches reading order.
        page_adds.sort(key=lambda a: (a["y0"], a["x0"]))
        for a in page_adds:
            x0 = max(0, int(a["x0"]) - BOARD_CROP_PAD)
            y0 = max(0, int(a["y0"]) - BOARD_CROP_PAD)
            x1 = min(w, int(a["x1"]) + BOARD_CROP_PAD)
            y1 = min(h, int(a["y1"]) + BOARD_CROP_PAD)
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
                    source_board_idx=10**6 + added,  # placeholder; reindex below
                    stones=stones,
                    black_to_play=True,
                    crop_png=buf.tobytes() if ok else None,
                    status="unreviewed",
                    page_idx=page_idx,
                    bbox_idx=next_idx,
                )
                added += 1
                next_idx += 1
            except Exception as e:
                log.warning("patch apply: discretize failed page %d: %s",
                            page_idx, e)

    # Reindex source_board_idx contiguously over (page_idx, bbox_idx).
    all_problems = list_problems(user_id, source)
    all_problems.sort(key=lambda p: (
        p.get("page_idx") or 0, p.get("bbox_idx") or 0,
    ))
    udir = tsumego_dir(user_id)
    reindexed = 0
    for new_sbi, p in enumerate(all_problems):
        if p.get("source_board_idx") == new_sbi:
            continue
        jp = udir / f"{p['id']}.json"
        d = json.loads(jp.read_text())
        d["source_board_idx"] = new_sbi
        jp.write_text(json.dumps(d, indent=2))
        reindexed += 1

    state["phase"] = "done"
    save_state(state)
    return {"deleted": deleted, "added": added, "reindexed": reindexed}


def _problems_grouped_by_page(user_id: str, source: str) -> dict[int, list[dict]]:
    from .tsumego import list_problems
    out: dict[int, list[dict]] = {}
    for p in list_problems(user_id, source):
        pi = p.get("page_idx")
        if pi is None:
            continue
        out.setdefault(pi, []).append(p)
    return out
