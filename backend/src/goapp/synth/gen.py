"""Batch-generate synthetic Go-book pages into $GOAPP_DATA_DIR/data/synth_pages/.

Usage:
    uv --directory backend run python -m goapp.synth.gen --count 500

Each page produces two sibling files named <page_id>.png and <page_id>.json:
    <page_id>.png   — rendered (degraded) page image
    <page_id>.json  — {"lang","size","boards":[{"bbox","window","edges_on_board","stones":[...]}]}

Pages are generated in parallel across CPU cores. The dominant cost is
`degrade()` (numpy noise + gaussian blur on a 1000×1400 image, ~700 ms),
which runs single-threaded per page; fanning out across cores gives
near-linear speedup until I/O saturates.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import time
from multiprocessing import Pool
from pathlib import Path

from .degrade import degrade
from .page_compose import compose_page

from ..paths import SYNTH_PAGES_DIR as DEFAULT_OUT  # noqa: E402


def _generate_one(args: tuple[int, str, bool]) -> str:
    """Worker: render one page and write its png + json. Returns the
    page_id used so the parent can log it.

    Run as a top-level function so multiprocessing can pickle it."""
    seed, out_dir, do_degrade = args
    rng = random.Random(seed)
    page = compose_page(rng=rng)
    if do_degrade:
        page = degrade(page, rng)

    page_id = (
        f"synth_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
        f"_{secrets.token_hex(4)}"
    )
    out = Path(out_dir)
    page.image.save(out / f"{page_id}.png")
    (out / f"{page_id}.json").write_text(json.dumps(page.to_label()))
    return page_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200, help="number of pages to generate")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; each page gets seed+i for reproducibility")
    ap.add_argument("--no-degrade", action="store_true",
                    help="skip rotation/blur/noise pass (useful for debugging)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="parallel worker processes (default: all CPUs)")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    do_degrade = not args.no_degrade

    jobs = [(base_seed + i, str(out), do_degrade) for i in range(args.count)]
    workers = max(1, min(args.workers, args.count))

    t0 = time.time()
    print(f"generating {args.count} pages with {workers} workers → {out}")
    if workers == 1:
        for i, job in enumerate(jobs):
            _generate_one(job)
            _maybe_log(i + 1, args.count, t0)
    else:
        with Pool(processes=workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_generate_one, jobs, chunksize=8)):
                _maybe_log(i + 1, args.count, t0)

    print(f"wrote {args.count} pages to {out}")


def _maybe_log(done: int, total: int, t0: float) -> None:
    if done % 25 == 0 or done == total:
        dt = time.time() - t0
        print(f"  {done}/{total}  ({dt / done:.2f}s/page)")


if __name__ == "__main__":
    main()
