# Training guide

Instructions for retraining the board and stone detectors.

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
uv --directory backend run python -m goapp.synth.gen --count 1500
```

~5 minutes on CPU (image manipulation, not GPU-bound).

### 2. Train the stone detector (stones-only, on crops)

```bash
uv --directory backend run --extra ml python -m goapp.ml.stone_detect.train \
    --limit 1500 --epochs 30 --device mps
```

On MPS (M3 Max): ~20-30 minutes.
On RTX 4090 / A100 cloud: ~10-15 minutes.
On CPU (M2): ~3 hours — only as last resort.

Output: `~/data/go-app/models/stone_detector.pt`

**Sanity check after training**:
- Val set should land around **P>=0.96, R>=0.90, mAP50>=0.93**.
- Training run artifacts in `~/data/go-app/models/runs/stone_detector/`.

### 3. Train the board detector

```bash
uv --directory backend run --extra ml python -m goapp.ml.board_detect.train \
    --limit 500 --epochs 25
```

Output: `~/data/go-app/models/board_detector.pt`

### 4. Measure on the val dataset

```bash
uv --directory backend run python -m goapp.cli.compare_on_val \
    --val-dir ~/data/go-app/data/val/hm2 \
    --model ~/data/go-app/models/stone_detector.pt \
    --status accepted --verbose
```

Open `/testing/validate/hm2` in the frontend to inspect results visually.

## Troubleshooting

**"device=cpu" in log despite `--device mps`**: Ultralytics falls back silently when
MPS can't run an op. Check with Activity Monitor — GPU usage should be
non-zero.

**Training hangs or OOMs**: reduce `--limit` to 500 to get a quick-failing
smaller run; once the pipeline is proven, scale back up.

**Results look dramatically worse than expected**: inspect
`~/data/go-app/data/yolo_stones/` to verify the extracted crops
look sensible (e.g., not blank, stones are visible).
