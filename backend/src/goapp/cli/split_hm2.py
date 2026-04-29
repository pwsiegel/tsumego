"""One-off: split each hm2.pdf board into the two problems it actually contains.

The hm2 dataset packs two distinct tsumego per detected board (upper-left
and lower-right). Re-detecting won't help — YOLO sees one board, correctly.
This script clusters the saved stones, splits each problem in two, and
rewrites the collection in place. Both halves keep a reference to the
original page crop (so "view original" still shows context) — only the
stone list and SGF differ.

Usage:
    python -m goapp.cli.split_hm2                # dry run, prints plan
    python -m goapp.cli.split_hm2 --apply        # write + delete originals
    python -m goapp.cli.split_hm2 --user-id <id> # default: 'local'

Algorithm: single-link cluster stones with Chebyshev distance <= 3, take
the two largest clusters as anchors (UL = lower row+col centroid, LR =
higher), then attach every other stone to whichever anchor has the
nearest stone. Validated to split all 178 saved hm2 boards into two
non-trivial groups.
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

from ..paths import tsumego_dir
from ..tsumego import stones_to_sgf

SOURCE = "hm2.pdf"
CLUSTER_THR = 3


def _cluster(stones: list[dict], thr: int) -> list[list[int]]:
    """Single-link clusters of stone indices under Chebyshev distance `thr`."""
    n = len(stones)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            d = max(
                abs(stones[i]["row"] - stones[j]["row"]),
                abs(stones[i]["col"] - stones[j]["col"]),
            )
            if d <= thr:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _split_two(stones: list[dict]) -> tuple[list[int], list[int]] | None:
    """Return (ul_indices, lr_indices) or None if can't form two anchors."""
    clusters = sorted(_cluster(stones, CLUSTER_THR), key=len, reverse=True)
    if len(clusters) < 2:
        return None
    a, b = clusters[0], clusters[1]

    def centroid_sum(idxs: list[int]) -> float:
        return sum(stones[i]["row"] + stones[i]["col"] for i in idxs) / len(idxs)

    if centroid_sum(a) > centroid_sum(b):
        a, b = b, a  # a is UL
    ul, lr = set(a), set(b)

    for stray in clusters[2:]:
        for sidx in stray:
            s = stones[sidx]

            def nearest(group: set[int]) -> int:
                return min(
                    max(
                        abs(stones[i]["row"] - s["row"]),
                        abs(stones[i]["col"] - s["col"]),
                    )
                    for i in group
                )

            (ul if nearest(ul) <= nearest(lr) else lr).add(sidx)

    return sorted(ul), sorted(lr)


def _new_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"tsumego_{stamp}_{secrets.token_hex(3)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user-id", default="local")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write new files and delete originals (default: dry run)",
    )
    args = parser.parse_args()

    udir = tsumego_dir(args.user_id)
    if not udir.exists():
        print(f"no tsumego dir at {udir}")
        return 1

    originals: list[tuple[Path, dict]] = []
    for jp in sorted(udir.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("source") == SOURCE:
            originals.append((jp, d))

    if not originals:
        print(f"no problems with source={SOURCE!r} in {udir}")
        return 0

    # Sort by (page_idx, bbox_idx) so renumbering is stable.
    originals.sort(key=lambda x: (x[1].get("page_idx", 0), x[1].get("bbox_idx", 0)))

    plan: list[dict] = []
    failures: list[tuple[Path, str]] = []
    new_idx = 0
    for jp, d in originals:
        stones = d["stones"]
        split = _split_two(stones)
        if split is None:
            failures.append((jp, f"only {len(_cluster(stones, CLUSTER_THR))} cluster(s)"))
            continue
        ul, lr = split
        for half_name, idxs in (("UL", ul), ("LR", lr)):
            plan.append({
                "original_id": d["id"],
                "original_json": jp,
                "page_idx": d.get("page_idx"),
                "bbox_idx": d.get("bbox_idx"),
                "image": d.get("image"),  # shared between halves
                "uploaded_at": d.get("uploaded_at"),
                "black_to_play": d.get("black_to_play", True),
                "half": half_name,
                "new_source_board_idx": new_idx,
                "stones": [stones[i] for i in idxs],
            })
            new_idx += 1

    print(f"found {len(originals)} {SOURCE} boards → {len(plan)} split problems")
    if failures:
        print(f"WARNING: {len(failures)} boards failed to split:")
        for jp, why in failures:
            print(f"  {jp.name}: {why}")

    if not args.apply:
        # Show first few entries as sanity check.
        for entry in plan[:4]:
            cnt = len(entry["stones"])
            print(
                f"  page={entry['page_idx']} bbox={entry['bbox_idx']} {entry['half']}"
                f" → idx={entry['new_source_board_idx']} ({cnt} stones)"
            )
        print("dry run — pass --apply to write")
        return 0

    # Apply: write new triples first (each new problem is a fresh id;
    # both halves point at the original PNG by filename). Then delete
    # the originals' JSON + SGF (PNGs stay — they're now shared).
    for entry in plan:
        pid = _new_id()
        # secrets.token_hex inside _new_id is random, but the timestamp
        # is per-second — guard against collisions in a tight loop by
        # retrying if a clash slips through.
        while (udir / f"{pid}.json").exists():
            pid = _new_id()

        sgf = stones_to_sgf(
            entry["stones"],
            black_to_play=entry["black_to_play"],
            image_ref=f"./{entry['image']}" if entry["image"] else None,
        )
        (udir / f"{pid}.sgf").write_text(sgf)
        (udir / f"{pid}.json").write_text(json.dumps({
            "id": pid,
            "source": SOURCE,
            "uploaded_at": entry["uploaded_at"],
            "source_board_idx": entry["new_source_board_idx"],
            "page_idx": entry["page_idx"],
            "bbox_idx": entry["bbox_idx"],
            "black_to_play": entry["black_to_play"],
            "status": "unreviewed",
            "image": entry["image"],
            "stones": [
                {"col": int(s["col"]), "row": int(s["row"]), "color": str(s["color"])}
                for s in entry["stones"]
            ],
        }, indent=2))

    deleted = 0
    for jp, _ in originals:
        sgf_path = jp.with_suffix(".sgf")
        jp.unlink(missing_ok=True)
        sgf_path.unlink(missing_ok=True)
        deleted += 1

    print(f"wrote {len(plan)} new problems, deleted {deleted} originals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
