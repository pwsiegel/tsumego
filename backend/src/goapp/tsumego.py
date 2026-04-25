"""Save human-accepted tsumego problems as SGF + sidecar metadata.

Each accepted problem is three files under tsumego_dir(user_id):
    {id}.sgf    SGF with SZ[19], AB/AW setup stones, PL[B] (by default).
    {id}.json   Metadata: source PDF, upload timestamp, source board index,
                black-to-play flag, plus the stone list for downstream
                consumers that don't want to parse SGF.
    {id}.png    The original board crop, when available.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .paths import tsumego_dir


def _sgf_coord(col: int, row: int) -> str:
    """SGF coordinates: a=0 … s=18, column then row."""
    return chr(ord("a") + col) + chr(ord("a") + row)


def stones_to_sgf(
    stones: list[dict],
    black_to_play: bool = True,
    image_ref: str | None = None,
) -> str:
    """Build a minimal tsumego SGF from a list of setup stones.

    Each stone: {"col": int 0-18, "row": int 0-18, "color": "B"|"W"}.
    `image_ref`, if given, is written as a custom IM[] property pointing
    at the saved crop image (e.g. "./{id}.png"). SGF parsers ignore
    unknown properties, so this is safe to include.
    """
    black = [s for s in stones if s.get("color") == "B"]
    white = [s for s in stones if s.get("color") == "W"]
    parts = [
        "(;",
        "FF[4]", "GM[1]", "SZ[19]",
        f"PL[{'B' if black_to_play else 'W'}]",
    ]
    if image_ref:
        parts.append(f"IM[{image_ref}]")
    if black:
        parts.append(
            "AB"
            + "".join(f"[{_sgf_coord(s['col'], s['row'])}]" for s in black)
        )
    if white:
        parts.append(
            "AW"
            + "".join(f"[{_sgf_coord(s['col'], s['row'])}]" for s in white)
        )
    parts.append(")")
    return "".join(parts)


def _remove_existing(user_id: str, source: str, source_board_idx: int) -> None:
    """Clear any previously-saved problem for this (source, board_idx).
    Supports toggling a decision: if the user rejects, then navigates
    back and accepts, we replace the old rejected record instead of
    piling up duplicates."""
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return
    for mp in udir.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        if (d.get("source") == source
                and d.get("source_board_idx") == source_board_idx):
            sgf = mp.with_suffix(".sgf")
            png_name = d.get("image")
            mp.unlink(missing_ok=True)
            sgf.unlink(missing_ok=True)
            if png_name:
                (udir / png_name).unlink(missing_ok=True)


STATUSES = ("unreviewed", "accepted", "accepted_edited", "rejected")


def save_problem(
    user_id: str,
    source: str,
    uploaded_at: str,
    source_board_idx: int,
    stones: list[dict],
    black_to_play: bool = True,
    crop_png: bytes | None = None,
    status: str = "unreviewed",
    page_idx: int | None = None,
    bbox_idx: int | None = None,
) -> Path:
    """Persist a problem as SGF + metadata JSON (+ optional crop PNG).

    `status` is one of STATUSES. Every status gets saved to disk so the
    user has an audit trail of their decisions (and so they can
    re-review). (source, source_board_idx) is the uniqueness key:
    saving again replaces any previous record for that pair.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status!r}")
    udir = tsumego_dir(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    _remove_existing(user_id, source, source_board_idx)

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    pid = f"tsumego_{stamp}_{secrets.token_hex(3)}"

    image_filename = f"{pid}.png" if crop_png is not None else None
    sgf = stones_to_sgf(
        stones,
        black_to_play,
        image_ref=f"./{image_filename}" if image_filename else None,
    )
    sgf_path = udir / f"{pid}.sgf"
    meta_path = udir / f"{pid}.json"

    sgf_path.write_text(sgf)
    if crop_png is not None:
        (udir / image_filename).write_bytes(crop_png)
    meta_path.write_text(json.dumps({
        "id": pid,
        "source": source,
        "uploaded_at": uploaded_at,
        "source_board_idx": source_board_idx,
        "page_idx": page_idx,
        "bbox_idx": bbox_idx,
        "black_to_play": black_to_play,
        "status": status,
        "image": image_filename,
        "stones": [
            {"col": int(s["col"]), "row": int(s["row"]), "color": str(s["color"])}
            for s in stones
        ],
    }, indent=2))
    return sgf_path


def problem_exists(user_id: str, source: str, source_board_idx: int) -> bool:
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return False
    for mp in udir.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        if (d.get("source") == source
                and d.get("source_board_idx") == source_board_idx):
            return True
    return False


def load_problem(user_id: str, problem_id: str) -> dict | None:
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return None
    mp = udir / f"{problem_id}.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None


def list_problems(user_id: str, source: str) -> list[dict]:
    """Return every problem belonging to `source`, sorted by source_board_idx."""
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return []
    out: list[dict] = []
    for mp in udir.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("source") != source:
            continue
        out.append(d)
    out.sort(key=lambda d: d.get("source_board_idx", 0))
    return out


def update_problem(
    user_id: str,
    problem_id: str,
    status: str | None = None,
    stones: list[dict] | None = None,
    black_to_play: bool | None = None,
) -> dict | None:
    """Rewrite a problem's status / stones / black_to_play in place.
    Rewrites the SGF when stones or black_to_play change. Returns the
    updated metadata dict or None if no such problem."""
    meta = load_problem(user_id, problem_id)
    if meta is None:
        return None
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        meta["status"] = status
    sgf_dirty = False
    if stones is not None:
        meta["stones"] = [
            {"col": int(s["col"]), "row": int(s["row"]), "color": str(s["color"])}
            for s in stones
        ]
        sgf_dirty = True
    if black_to_play is not None:
        meta["black_to_play"] = black_to_play
        sgf_dirty = True
    udir = tsumego_dir(user_id)
    if sgf_dirty:
        image_filename = meta.get("image")
        new_sgf = stones_to_sgf(
            meta["stones"],
            meta.get("black_to_play", True),
            image_ref=f"./{image_filename}" if image_filename else None,
        )
        (udir / f"{problem_id}.sgf").write_text(new_sgf)
    (udir / f"{problem_id}.json").write_text(json.dumps(meta, indent=2))
    return meta


def list_collections(user_id: str) -> list[dict]:
    """Aggregate saved problems by source PDF.

    Returns a list of {source, count, last_uploaded_at}, one entry per
    distinct source PDF name, sorted by most recent upload first. Later
    this will back the home screen's "Collections" list.
    """
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return []
    groups: dict[str, dict] = {}
    for mp in udir.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        src = d.get("source")
        if not src:
            continue
        g = groups.setdefault(src, {
            "source": src,
            "count": 0,
            "last_uploaded_at": "",
            "accepted": 0,
            "accepted_edited": 0,
            "rejected": 0,
            "unreviewed": 0,
        })
        g["count"] += 1
        status = d.get("status", "unreviewed")
        if status in g:
            g[status] += 1
        ts = d.get("uploaded_at", "")
        if ts > g["last_uploaded_at"]:
            g["last_uploaded_at"] = ts
    out = list(groups.values())
    out.sort(key=lambda g: g["last_uploaded_at"], reverse=True)
    return out


def delete_problem(user_id: str, problem_id: str) -> bool:
    """Delete one problem's SGF + JSON + image PNG. Returns True if found."""
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return False
    mp = udir / f"{problem_id}.json"
    if not mp.exists():
        return False
    try:
        d = json.loads(mp.read_text())
    except json.JSONDecodeError:
        d = {}
    image = d.get("image")
    mp.unlink(missing_ok=True)
    (udir / f"{problem_id}.sgf").unlink(missing_ok=True)
    if image:
        (udir / image).unlink(missing_ok=True)
    return True


def delete_collection(user_id: str, source: str) -> int:
    """Delete every SGF + metadata pair whose metadata source matches.

    Returns the number of problems removed.
    """
    udir = tsumego_dir(user_id)
    if not udir.exists():
        return 0
    removed = 0
    for mp in udir.glob("*.json"):
        try:
            d = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("source") != source:
            continue
        sgf = mp.with_suffix(".sgf")
        mp.unlink(missing_ok=True)
        sgf.unlink(missing_ok=True)
        removed += 1
    return removed
