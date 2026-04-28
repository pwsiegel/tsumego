"""Local PDF ingest CLI: detect boards on the laptop, then push to GCS.

Cloud Run does CPU-only ONNX inference and chews minutes per book; running
this locally is much faster (and free). Output is identical to a
server-side ingest, so the cloud app can't tell the difference.

Usage:
    python -m goapp.cli.ingest_pdf --pdf book.pdf --user pwsiegel@gmail.com

If --user looks like an email it's hashed via `_hash_email` to match the
production IAP-derived user_id. Otherwise it's treated as the raw id.
After detection completes, files under
$GOAPP_DATA_DIR/data/tsumego/{user_id}/ are rsynced to the GCS bucket
(--no-upload to skip).

Page-level detection is fanned out across `--workers` processes via
ProcessPoolExecutor. Saves still happen serially in the main process
because `source_board_idx` must be assigned in page-then-bbox order.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..auth import _hash_email
from ..paths import tsumego_dir
from ..tsumego import problem_exists, save_problem


def resolve_user_id(user: str) -> str:
    if "@" in user:
        return _hash_email(user.strip().lower())
    return user


# BLAS/OpenMP default to all-cores; with N workers each running its own
# multi-threaded ONNX session, we'd badly oversubscribe the CPU. Force
# single-threaded numerics inside each worker so N workers ≈ N cores.
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
    from ..ml.board_detect.detect import detect_boards_yolo
    from ..ml.pipeline import BOARD_CROP_PAD, discretize_crop

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


def _count_pages(pdf_path: Path) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    return len(pdf)


def main() -> int:
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser = argparse.ArgumentParser(
        description="Run board detection on a PDF locally and sync results to GCS.",
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument(
        "--user", required=True,
        help="email (gets hashed) or raw user_id",
    )
    parser.add_argument(
        "--source", default=None,
        help="source name to record (defaults to PDF filename)",
    )
    parser.add_argument("--bucket", default="tsumego-pwsiegel-data")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"page-level worker processes (default: {default_workers})",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"error: pdf not found: {args.pdf}", file=sys.stderr)
        return 1

    user_id = resolve_user_id(args.user)
    source = args.source or args.pdf.name
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pdf_path = str(args.pdf.resolve())
    total_pages = _count_pages(args.pdf)

    print(f"==> ingesting {args.pdf.name} as user_id={user_id} source={source!r}")
    print(f"  pages: {total_pages}, workers: {args.workers}")

    page_results: dict[int, list[dict]] = {}
    t_start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_worker_init,
    ) as ex:
        futures = {
            ex.submit(_process_page, pdf_path, i): i
            for i in range(total_pages)
        }
        done = 0
        for fut in as_completed(futures):
            page_idx, boards = fut.result()
            page_results[page_idx] = boards
            done += 1
            print(f"  [{done}/{total_pages}] page {page_idx + 1}: {len(boards)} boards")
    print(f"  detection took {time.time() - t_start:.1f}s")

    total_saved = 0
    skipped = 0
    failed = 0
    source_board_idx = 0
    for page_idx in range(total_pages):
        for board in page_results.get(page_idx, []):
            if "error" in board:
                print(f"  ! page {page_idx + 1} bbox {board['bbox_idx']}: "
                      f"{board['error']}", file=sys.stderr)
                failed += 1
                source_board_idx += 1
                continue
            if problem_exists(user_id, source, source_board_idx):
                skipped += 1
                source_board_idx += 1
                continue
            save_problem(
                user_id=user_id,
                source=source,
                uploaded_at=uploaded_at,
                source_board_idx=source_board_idx,
                stones=board["stones"],
                black_to_play=True,
                crop_png=board["crop_png"],
                status="unreviewed",
                page_idx=page_idx,
                bbox_idx=board["bbox_idx"],
            )
            total_saved += 1
            source_board_idx += 1

    print(f"==> detection done: {total_saved} saved, "
          f"{skipped} skipped, {failed} failed")

    if args.no_upload:
        print("==> skipping upload (--no-upload)")
        return 0

    udir = tsumego_dir(user_id)
    if not udir.exists():
        print(f"==> nothing to upload (no problems for {user_id})")
        return 0

    dst = f"gs://{args.bucket}/data/tsumego/{user_id}/"
    print(f"==> syncing {udir}/ -> {dst}")
    rc = subprocess.call(["gsutil", "-m", "rsync", "-r", f"{udir}/", dst])
    if rc != 0:
        print(f"error: gsutil rsync exited {rc}", file=sys.stderr)
        return rc
    print("==> upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
