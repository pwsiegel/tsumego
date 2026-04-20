"""Batch-generate synthetic Go-book pages into training_data/synth_pages/.

Usage:
    uv --directory api run python -m goapp_api.synth.gen --count 500

Each page produces two sibling files named <page_id>.png and <page_id>.json:
    <page_id>.png   — rendered (degraded) page image
    <page_id>.json  — {"lang","size","boards":[{"bbox","window","edges_on_board","stones":[...]}]}
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from pathlib import Path

from .degrade import degrade
from .page_compose import compose_page

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "training_data" / "synth_pages"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200, help="number of pages to generate")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; each page gets seed+i for reproducibility")
    ap.add_argument("--no-degrade", action="store_true",
                    help="skip rotation/blur/noise pass (useful for debugging)")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)

    t0 = time.time()
    for i in range(args.count):
        rng = random.Random(base_seed + i)
        page = compose_page(rng=rng)
        if not args.no_degrade:
            page = degrade(page, rng)

        page_id = f"synth_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{secrets.token_hex(4)}"
        png_path = out / f"{page_id}.png"
        json_path = out / f"{page_id}.json"
        page.image.save(png_path)
        json_path.write_text(json.dumps(page.to_label()))

        if (i + 1) % 25 == 0 or i == args.count - 1:
            dt = time.time() - t0
            print(f"  {i + 1}/{args.count}  ({dt / (i + 1):.2f}s/page)")

    print(f"wrote {args.count} pages to {out}")


if __name__ == "__main__":
    main()
