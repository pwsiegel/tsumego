"""Validation dataset endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

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


def _process_problem(val_dir, entry) -> dict:
    """Run the live pipeline on one validation entry and shape the
    per-problem result the way the web view expects."""
    import json as _json
    import cv2
    from ...cli.compare_on_val import _run_pipeline

    stem = entry["stem"]
    meta_path = val_dir / "metadata" / f"{stem}.json"
    if not meta_path.exists():
        return {"stem": stem, "status": "error", "error": "metadata not found"}
    meta = _json.loads(meta_path.read_text())
    gt = {(s["col"], s["row"], s["color"]) for s in meta.get("stones", [])}

    img_path = val_dir / "images" / f"{stem}.png"
    crop = cv2.imread(str(img_path))
    if crop is None:
        return {"stem": stem, "status": "error", "error": "image not found"}

    try:
        pred = _run_pipeline(crop, peak_thresh=0.3)
    except Exception as e:
        return {"stem": stem, "status": "error", "error": str(e)}

    if pred == gt:
        return {"stem": stem, "status": "exact"}

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

    return {
        "stem": stem,
        "status": "changed",
        "gt_count": len(gt),
        "pred_count": len(pred),
        "missed": missed,
        "extra": extra,
        "flips": flips,
        "gt_stones": gt_stones,
        "pred_stones": pred_stones,
    }


VAL_WORKERS = 4  # parallelism for per-problem pipeline runs


@router.get("/{dataset}/run")
def val_run_endpoint(dataset: str, status: str = "accepted") -> StreamingResponse:
    """Stream pipeline results as NDJSON so the web view can render a
    progress bar that ticks per problem.

    Pipeline runs are fanned out across a small thread pool —
    discretize_crop is GIL-light (cv2 + ultralytics inference release
    the GIL), so threads give real speedup without the model-load
    cost of separate processes. Events arrive in completion order,
    not submission order; each carries its own stem.

    Wire format (one JSON object per line):
      {"event":"start","total":N,"dataset":...,"filter_status":...}
      {"event":"problem","result":{...}}    # one per problem
      {"event":"done","exact":N,"changed":N,"errors":N}
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    val_dir = DATA_DIR / "val" / dataset
    manifest_path = val_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"no manifest in val/{dataset}")

    manifest = _json.loads(manifest_path.read_text())
    if status == "all":
        entries = manifest["problems"]
    else:
        entries = [p for p in manifest["problems"] if p["status"] == status]

    def generate():
        yield _json.dumps({
            "event": "start",
            "total": len(entries),
            "dataset": dataset,
            "filter_status": status,
        }) + "\n"
        exact = changed = errors = 0
        workers = max(1, min(VAL_WORKERS, len(entries)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_problem, val_dir, e) for e in entries]
            for fut in as_completed(futures):
                result = fut.result()
                if result["status"] == "exact":
                    exact += 1
                elif result["status"] == "changed":
                    changed += 1
                else:
                    errors += 1
                yield _json.dumps({"event": "problem", "result": result}) + "\n"
        yield _json.dumps({
            "event": "done",
            "exact": exact, "changed": changed, "errors": errors,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
