"""Validation dataset endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...paths import DATA_DIR

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/val", tags=["val"])


@router.post("/{dataset}/problems/{stem}/stones")
def val_update_stones_endpoint(dataset: str, stem: str, req: dict) -> dict:
    """Overwrite the ground-truth stones for a problem in a val dataset."""
    import json as _json
    import time
    from ...tsumego import stones_to_sgf

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


@router.get("/{dataset}/gt-edits")
def val_gt_edits_endpoint(dataset: str) -> Response:
    """Chronological log of ground-truth edits."""
    import json as _json
    path = DATA_DIR / "val" / dataset / "gt_edits.json"
    if not path.exists():
        return Response(content=_json.dumps([]), media_type="application/json")
    return Response(content=path.read_bytes(), media_type="application/json")


@router.get("/{dataset}/comparison")
def val_comparison_endpoint(dataset: str) -> Response:
    """Serve the comparison JSON."""
    path = DATA_DIR / "val" / dataset / "comparison.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no comparison.json in val/{dataset}")
    return Response(content=path.read_bytes(), media_type="application/json")


@router.get("/{dataset}/images/{stem}.png")
def val_image_endpoint(dataset: str, stem: str) -> Response:
    path = DATA_DIR / "val" / dataset / "images" / f"{stem}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"image not found: {stem}")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/{dataset}/run")
def val_run_endpoint(dataset: str, status: str = "accepted") -> Response:
    """Run the live pipeline against every problem in the validation set."""
    import json as _json
    import cv2
    from ...cli.compare_on_val import _run_pipeline

    val_dir = DATA_DIR / "val" / dataset
    manifest_path = val_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"no manifest in val/{dataset}")

    manifest = _json.loads(manifest_path.read_text())
    if status == "all":
        entries = manifest["problems"]
    else:
        entries = [p for p in manifest["problems"] if p["status"] == status]

    results: list[dict] = []
    exact = 0
    for entry in entries:
        stem = entry["stem"]
        meta_path = val_dir / "metadata" / f"{stem}.json"
        if not meta_path.exists():
            results.append({"stem": stem, "status": "error", "error": "metadata not found"})
            continue
        meta = _json.loads(meta_path.read_text())
        gt = {(s["col"], s["row"], s["color"]) for s in meta.get("stones", [])}

        img_path = val_dir / "images" / f"{stem}.png"
        crop = cv2.imread(str(img_path))
        if crop is None:
            results.append({"stem": stem, "status": "error", "error": "image not found"})
            continue

        try:
            pred = _run_pipeline(crop, peak_thresh=0.3)
        except Exception as e:
            results.append({"stem": stem, "status": "error", "error": str(e)})
            continue

        if pred == gt:
            exact += 1
            results.append({"stem": stem, "status": "exact"})
        else:
            missed = gt - pred
            extra = pred - gt
            gt_pos = {(c, r): col for (c, r, col) in gt}
            pred_pos = {(c, r): col for (c, r, col) in pred}
            flips = [
                {"col": c, "row": r, "gt_color": gt_pos[(c, r)], "pred_color": pred_pos[(c, r)]}
                for (c, r) in gt_pos.keys() & pred_pos.keys()
                if gt_pos[(c, r)] != pred_pos[(c, r)]
            ]
            flip_positions = {(f["col"], f["row"]) for f in flips}
            missed = [{"col": c, "row": r, "color": col}
                      for (c, r, col) in sorted(missed) if (c, r) not in flip_positions]
            extra = [{"col": c, "row": r, "color": col}
                     for (c, r, col) in sorted(extra) if (c, r) not in flip_positions]

            gt_stones = [{"col": s["col"], "row": s["row"], "color": s["color"]}
                         for s in meta.get("stones", [])]
            pred_stones = [{"col": c, "row": r, "color": col}
                           for (c, r, col) in sorted(pred)]

            results.append({
                "stem": stem,
                "status": "changed",
                "gt_count": len(gt),
                "pred_count": len(pred),
                "missed": missed,
                "extra": extra,
                "flips": flips,
                "gt_stones": gt_stones,
                "pred_stones": pred_stones,
            })

    body = {
        "dataset": dataset,
        "filter_status": status,
        "total": len(entries),
        "exact": exact,
        "changed": sum(1 for r in results if r["status"] == "changed"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "problems": results,
    }
    return Response(content=_json.dumps(body), media_type="application/json")
