# Training plan — stones v2 + corner detector

Ordered instructions for the next training cycle, to be executed on a
machine with MPS or CUDA. The Mac mini with M2 CPU-only is too slow for
practical iteration; a laptop with an M3 Max or a rented cloud GPU
(RTX 4090 / A100 class) is the intended target.

## Why we're retraining

**Stone detector (retrain)** — the current `stone_detector.pt` was trained on
synth *pages*, but at inference it runs on per-board *crops*. Training-
distribution mismatch. It still works, but we suspect hoshi false positives
and some Korean-character false positives are partly due to this. New
training uses per-board crops; hoshi dots are visible in every training
crop but deliberately unlabeled, so YOLO learns to treat them as background.

**Corner detector (new)** — 4-class YOLO bbox detector (TL/TR/BL/BR).
Corners give exact grid geometry (`pitch_x = (TR − TL).x / 18`, etc.)
without relying on classical 1D density-profile scanning, which has been
the source of several reliability bugs (board 100 snap drift, board 108
catastrophic pitch estimation failure). When reliable, corners replace
most of what `edge_inference.py` + `pitch.py` do.

Classical edge detection stays as a fallback for crops where the corner
detector finds fewer than 2 corners (rare — middle-strip crops, or
failure cases).

## Prerequisites

- Python env with `uv sync --extra ml` run (pulls in `ultralytics`, `torch`)
- `~/data/go-app/` with at least:
  - `data/synth_pages/` — 1500+ rendered synth pages. If empty, regenerate
    (see step 1).
  - `models/` — directory will be populated by training

Set `GOAPP_DATA_DIR` if data isn't at `~/data/go-app`:
```bash
export GOAPP_DATA_DIR=/path/to/data-dir
```

## Steps

### 1. (If needed) Regenerate synth pages

Skip if `~/data/go-app/data/synth_pages/` already has ~1500 pages.
Otherwise:

```bash
uv --directory api run python -m goapp_api.synth.gen --count 1500
```

~5 minutes on CPU (image manipulation, not GPU-bound).

### 2. Train the stone detector (stones-only, on crops)

```bash
uv --directory api run --extra ml python -m goapp_api.train_stones_yolo \
    --limit 1500 --epochs 30 --device mps
```

On MPS (M3 Max): ~20-30 minutes.
On RTX 4090 / A100 cloud: ~10-15 minutes.
On CPU (M2): ~3 hours — only as last resort.

Output: `~/data/go-app/models/stone_detector_yolo.pt`

**Sanity check after training**:
- Val set should land around **P≥0.96, R≥0.90, mAP50≥0.93**. Anything lower
  than that is a regression from the last run.
- Training run artifacts in `~/data/go-app/models/runs/stone_detector/`.

### 3. Train the corner detector

```bash
uv --directory api run --extra ml python -m goapp_api.train_corners_yolo \
    --limit 1500 --epochs 20 --device mps
```

On MPS (M3 Max): ~15-20 minutes.

Output: `~/data/go-app/models/corner_detector.pt`

**Sanity check**: corners are an easier task than stones (fewer, more
distinctive patterns); expect convergence to **mAP50 > 0.95** within 10-15
epochs.

### 4. Swap in new stone weights

The new stones model lands at `stone_detector_yolo.pt`. The live path the
server loads is `stone_detector.pt`. After training completes and the val
metrics look right:

```bash
cp ~/data/go-app/models/stone_detector_yolo.pt \
   ~/data/go-app/models/stone_detector.pt
```

Keep a dated backup of the old weights if you want to compare:
```bash
cp ~/data/go-app/models/stone_detector.pt \
   ~/data/go-app/models/stone_detector_v1_$(date +%Y%m%d).pt
```

### 5. (TODO, not in this commit) Wire in corner-based geometry resolver

Post-training work once weights exist. A new `corner_inference.py` module
(mirroring `stone_inference.py`) loads the corner model and runs it on
a crop. A new `geometry_resolver.py` consumes corner detections + edge
fallback and emits `(pitch_x, pitch_y, origin_x, origin_y, col_min,
row_min, visible_cols, visible_rows, edges)` — replacing the current
~40-line edge/pitch ladder in `main.py`'s `_discretize_board`.

Interface sketch:

```python
def resolve_geometry(
    crop_bgr: np.ndarray,
    corner_detections: list[dict],  # [{corner: 'tl'|'tr'|'bl'|'br', x, y, conf}]
) -> GridGeometry:
    if len(corner_detections) >= 4:
        return _from_four_corners(corner_detections)
    if len(corner_detections) >= 2:
        return _from_two_or_three_corners(corner_detections, crop_bgr)
    if len(corner_detections) == 1:
        return _from_single_corner_plus_edges(corner_detections[0], crop_bgr)
    return _classical_fallback(crop_bgr)   # current edge_inference + pitch logic
```

### 6. Measure on the val dataset

Once integrated, regenerate the comparison JSON against the new stone
weights and the new geometry resolver:

```bash
uv --directory api run python -m goapp_api.generate_comparison \
    --val-dir ~/data/go-app/data/val/hm2 \
    --old-model ~/data/go-app/models/stone_detector_v1_YYYYMMDD.pt \
    --new-model ~/data/go-app/models/stone_detector.pt \
    --out ~/data/go-app/data/val/hm2/comparison.json \
    --status all
```

Open `/compare/hm2` in the frontend to eyeball results. Key metrics:

- Per-problem exact match on `accepted` subset: should stay near 95%
  (improvement over v1's 95.2%)
- Per-problem exact match on `accepted_edited` subset: should improve
  meaningfully over v1's 14.8%
- Total ground-truth match: should exceed v1's 70.2%

## Troubleshooting

**"device=cpu" in log despite `--device mps`**: Ultralytics falls back silently when
MPS can't run an op. Check with Activity Monitor → GPU usage should be
non-zero. If it's zero throughout, MPS isn't actually being used; may
need to upgrade `ultralytics` in `pyproject.toml`.

**Training hangs or OOMs**: reduce `--limit` to 500 to get a quick-failing
smaller run; once the pipeline is proven, scale back up.

**Results look dramatically worse than expected**: compare the val numbers
against the expected values above. If much lower, inspect
`~/data/go-app/data/yolo_{stones|corners}/` to verify the extracted crops
look sensible (e.g., not blank, stones are visible).

## What's in this commit

- `train_stones_yolo.py` — rewritten to train on per-board crops (was pages)
- `train_corners_yolo.py` — already rewritten earlier to train on crops; added
  `--device` arg
- This document

## What's NOT in this commit (follow-on work)

- `corner_inference.py` module
- `geometry_resolver.py` module
- `main.py` integration to use corner detector + resolver instead of
  classical edge/pitch ladder
- Retired code in `edge_inference.py` / `pitch.py` (kept as fallback; can
  trim once corners are proven)
