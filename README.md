# Go Problem Workbook

Extract Go problems from scanned PDFs of tsumego books, build a per-user library,
and (eventually) train against them with engine assistance.

The pipeline ingests a PDF page-by-page, finds the board diagrams, reads off the
stones, and saves each problem as SGF + metadata + the original board crop.

## What it can do

### Frontend (`web/`, React + TypeScript + Vite)

User-facing pages:

| Route | What it does |
|---|---|
| `/` | Landing page; lists previously-ingested PDF collections. |
| `/upload` | Upload a PDF. Streams ingest progress (page rendered → board saved). |
| `/collections/:source` | Grid of every problem extracted from one PDF. |
| `/collections/:source/review` | Sequentially review unreviewed problems; accept / reject / edit. |
| `/collections/:source/problem/:id` | Single problem: SGF view, edit stones, change status. |

Developer tools (under `/testing/…`):

| Route | What it does |
|---|---|
| `/testing/bbox` | Visualize YOLO board-bbox detection on each PDF page. |
| `/testing/stones` | Visualize stone detection + grid discretization for one board. |
| `/testing/validate/:dataset` | Run the full pipeline against a labeled validation set; inspect mistakes side-by-side. |
| `/compare/:dataset` | Side-by-side compare of detector output vs. ground truth. |

A `HealthGate` covers the app on cold start and polls `/api/health` until
the YOLO models have finished warming, so the first request never lands
mid-warmup.

### Backend (`backend/`, FastAPI + ultralytics + opencv)

| Group | Endpoints (selected) | Purpose |
|---|---|---|
| `/api/health` | `GET /api/health` | Liveness + model-warm status (drives the cold-start gate). |
| `/api/pdf` | `POST /upload-url`, `POST /ingest-from-upload`, `POST /ingest`, `GET /boards`, `GET /board-crop/...`, `GET /board-discretize/...` | PDF upload + ingest. In cloud, `upload-url` returns a signed PUT to GCS to bypass Cloud Run's 32 MiB request cap; `ingest-from-upload` then streams NDJSON progress (`page_rendered`, `board_saved`, `done`) as it processes. The `bbox-*` and `boards` endpoints back the developer tools. |
| `/api/tsumego` | `GET /collections`, `POST /save`, `GET /{id}`, `POST /{id}/status`, `DELETE /{id}` | Per-user library CRUD. Storage is plain files (SGF + JSON sidecar + PNG crop) under `data/tsumego/{user_id}/`. |
| `/api/val` | `GET /{dataset}/comparison`, `POST /{dataset}/problems/{stem}/stones`, `GET /{dataset}/run` | Validation tooling for the developer pages. |

Identity comes from IAP (header `X-Goog-Authenticated-User-Email`) in the
cloud and a constant `local-dev` user locally.

## How it works

### Models (trained)

Both are YOLOv8-nano fine-tunes (ultralytics). They're trained on
synthetic data only — no labeled real-world scans needed.

- **Board detector** (`backend/data/models/board_detector.pt`).
  Single class `board`. Input: a full PDF page rendered at 2× scale.
  Output: bounding boxes around every Go diagram on the page.
  Trained on whole synthetic pages.

- **Stone detector** (`backend/data/models/stone_detector.pt`).
  Two classes: `B` (black), `W` (white). Input: a single board crop
  (output of the board detector, padded by 10 px). Output: bbox per
  stone; we take centers. Trained on per-board crops cut out of the
  same synthetic pages.

Hoshi (star points) are rendered into the synthetic boards but
**not labeled** as stones — the model learns to treat them as background.

### Classical algorithms (no training)

These run after stone detection to convert pixel positions into a 19×19
board state. Living under `backend/src/goapp/ml/` even though they aren't
ML — kept there as part of the detection pipeline.

- **Edge detection** (`edge_detect/`). For each side of a board crop,
  decides whether that side is a real frame line (the outer 19th line)
  or an interior grid row/column near the crop edge. Method: adaptive
  threshold → 1D dark-pixel-density profile across a strip near that
  side → score the outermost peak as frame-like vs. grid-like (uses
  thickness-ratio and darkness-delta vs. the next interior peak).

- **Pitch measurement** (`pitch/`). Measures the grid cell size in pixels
  by finding the distance between the frame line and the first interior
  grid line on each confirmed-frame side, then takes the median. Works
  directly off the printed grid, not off detected stones.

- **Discretization** (`discretize/`). Given stone centers (from the
  stone CNN) plus the 4-bit edge classifier output, computes:
  `cell_size` (25th percentile of pairwise stone distances within a
  plausible pitch range — a robust nearest-neighbor estimate),
  `origin_x/y` (brute-force search in `[0, cell_size)` minimizing snap
  residual), and the position of the visible window inside the full
  19×19 board. Edge flags disambiguate which corner/edge of the board
  the crop covers.

The pipeline (`ml/pipeline.py`) ties these together: YOLO board → crop →
stone YOLO + classical geometry → discretized 19×19 position.

### Synthetic data (`backend/src/goapp/synth/`)

Generated pages are the only training input.

- `board_render.py` — draws Go boards (random size cropping, hoshi,
  stones randomly placed, optional move-number labels, multiple board
  themes).
- `page_compose.py` — lays one or more boards on a page with random text
  blocks, in multiple languages (mimics tsumego book pages).
- `degrade.py` — simulates scan artifacts: blur, noise, JPEG-compression
  artifacts, slight rotation, off-white paper tint.
- `gen.py` — entry point. Each generated page produces a `.png` and a
  `.json` (board bboxes + per-stone positions + edge flags).

Defaults to 1500 pages (`make synth` runs `--count 1500`). Generation is
CPU-bound, no GPU needed; ~5 minutes on a recent Mac.

## Repo layout

```
tsumego/
├── web/                   React + TS + Vite frontend
├── backend/
│   ├── data/models/       Trained .pt weights baked into the serving image
│   └── src/goapp/
│       ├── api/           FastAPI routers (health, pdf, tsumego, val)
│       ├── ml/            Detection + classical-geometry pipeline
│       ├── cli/           Validation runner, dataset export, comparison
│       └── synth/         Synthetic page generator
├── training/              Vertex AI training image + job specs
├── Dockerfile             Serving image (Cloud Run)
├── Makefile               All common tasks
└── docker-compose.yml     Local containerized dev
```

## Getting started

### Prerequisites

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **Node 18+**

The trained weights live in the repo at `backend/data/models/{board,stone}_detector.pt`
(~12 MB total, committed to git), so a fresh clone runs out of the box —
no extra setup, no GCS pull. To retrain, see "Train models" below.

(Optional, training only) **Local data dir**, default `~/data/go-app/`:
```
~/data/go-app/
└── data/
    ├── synth_pages/    Generated training pages (PNG + JSON)
    ├── val/<dataset>/  Labeled validation sets, if you have them
    └── training_runs/  Ultralytics run artifacts (logs, intermediate checkpoints)
```
Override the location with `GOAPP_DATA_DIR=/path/to/dir`.

### First-time setup

```bash
make setup    # uv sync --extra ml --extra dev + npm install
```

### Run the app

Two terminals:

```bash
make api      # backend on http://localhost:8001
make web      # frontend on http://localhost:5173 (proxies /api → :8001)
```

Or one container:

```bash
make docker-up  # frontend + backend on http://localhost:8080
```

## Common tasks

All driven by `make help`.

### Generate synthetic training data

```bash
make synth                       # writes 1500 pages to ~/data/go-app/data/synth_pages/
```

### Train models locally (Mac M-series)

```bash
make train-boards                # ~10 min on MPS
make train-stones                # ~20-30 min on MPS; set DEVICE=cpu if no GPU
```

Outputs overwrite `backend/data/models/{board,stone}_detector.pt` directly.
Commit the new weights if you want them to ship with the next serving build.

### Train models on Vertex AI (L4 spot, ~$0.15-0.30 per run)

Requires GCP setup (Cloud Run / GCS / IAP already provisioned — see
`training/job-*.yaml` for project + bucket names).

```bash
make sync-synth                  # local synth_pages → gs://...-data/data/synth_pages
make build-training-image        # one-time per Dockerfile change
make train-cloud-boards          # or train-cloud-stones; submits job and exits
# … watch progress at https://console.cloud.google.com/vertex-ai/training
make pull-weights                # GCS → backend/data/models/
make deploy                      # rebuild + roll out serving image with new weights
```

The training entrypoint reads `synth_pages` from the GCS FUSE mount and
keeps derived YOLO datasets / ultralytics run artifacts on local
container disk (FUSE writes are slow). Only the final `best.pt` is
copied back to the bucket.

### Validate against a labeled dataset

```bash
make validate DATASET=hm2        # prints per-problem comparison report
```

Open `/testing/validate/hm2` in the frontend for a side-by-side visual
of detector output vs. ground truth.

### Deploy to Cloud Run

```bash
make deploy                      # Cloud Build → gcr → Cloud Run rollout
make logs                        # last 50 lines of Cloud Run logs
```

## Roadmap

Tracks ideas as they come up. Ordering within a section is rough priority.

### Play mode
- [ ] **Under-the-stones moves** — allow placing a move on an intersection that already holds a stone, provided the prior stone has been captured in the evolving position. Requires capture/liberty logic.
- [ ] **Multi-number annotation** — a single intersection may hold several move numbers over the life of a problem (move N played there, captured, move M played there later). Render stacked or comma-separated labels.
- [ ] **Illegal-move detection** — warn on suicide / ko violation / playing on an occupied point that hasn't been captured.
- [ ] Branching variations (tree of moves, not just a linear sequence).

### PDF import / detection
- [ ] Auto-detect corner placement (row/column intensity peaks to pre-place the 4 handles).
- [ ] **Correction UI**: click a detected stone to cycle Empty → B → W → Empty. Also serves as the labeling tool.
- [ ] Export labeled patches as training data (JSON + image tiles).
- [ ] Tiny CNN patch classifier trained in PyTorch, exported to ONNX, loaded via onnxruntime-web.
- [ ] Batch-ingest an entire PDF in one call (backend endpoint returning all detected problems).
- [ ] **Problem-label detection (OCR)** — crop a region around each detected board and OCR out the original book's label ("problem 7", "問題 3", "第 12 題", etc.) so imported problems keep their original numbering instead of sequential indices. Multilingual Tesseract is the obvious starting point, but accuracy craters on low-DPI photocopy scans — needs a fallback and probably per-book heuristics for where the label sits relative to the board.
- [ ] **Stone-annotation detection** — books frequently print a number, triangle/square/circle mark, or a letter/kanji inside a stone to identify "move 7" / "the marked stone" / "stone A". Detecting these would let us (a) transcribe the book's annotations accurately into the SGF (`LB[]`, `TR[]`, numbered move sequences) rather than collapsing everything to a setup position, and (b) disambiguate stones from nearby printed characters — observed on hm2 where a Korean character near the board was misread as a white stone by the detector. Plausible approach: after stone detection, crop each stone's interior and run a small CNN or OCR pass for marks/digits/characters; also use the presence of detected characters outside stones as a negative signal to suppress near-board text false positives.

### Solving & analysis

The obvious primitive is KataGo — strongest open-source Go engine, MIT-licensed, runs locally. The non-obvious problem: KataGo evaluates the *whole board*. Drop a corner tsumego onto an otherwise empty 19×19 and its policy will open a different corner, because that's the best move on this global position. Making it useful for local problems is real work.

Candidate approaches (not committed to one yet):

- [ ] **Local fencing.** Wall off the tactical region with thick stones on the owner's side; engine attention concentrates locally. Cheapest to prototype; breaks on problems where outside influence matters.
- [ ] **Local search + KataGo as evaluator.** Don't trust KataGo's policy; use its *value* head only. Generate candidate moves within a bounded active region, alpha-beta or MCTS over local moves, KataGo scores terminal positions. How real tsumego solvers work. More effort, more robust.
- [ ] **Fine-tune on tsumego.** Continue-train KataGo (or a smaller net) on a labeled tsumego dataset (~few hundred thousand problems exist publicly). Net learns "in problem-like positions, answer locally." Most ML work; generalizes best across problem types.
- [ ] **Dedicated tsumego solvers (prior art).** Proof-number search, GnuGo's life/death reader, various academic tools. Usually narrower in scope. Worth surveying before reinventing.

Downstream tasks that depend on picking one:

- [ ] **Check submitted solutions** — given an ingested problem, detect the intended outcome (black lives / black kills / connects / …) and evaluate whether a submitted sequence achieves it.
- [ ] **Auto-generate solutions** — principal variation from the solver, rendered as numbered moves.
- [ ] **Problem categorization** — classify by type (life-and-death, tesuji, connection, endgame) and by motif (ladder, net, throw-in, snapback…). No off-the-shelf tool exists. The global-vs-local concern is lighter here since categorization cares about *shape*, not best-move. Approach: embed the position with a pretrained Go net, cluster, hand-label the clusters. Becomes a tag taxonomy on each ingested problem.

Open infra question: bundle KataGo (adds ~100MB and a C++ dependency) vs. run it as a separate service.

### Persistence / export
- [ ] Save/load individual problems (localStorage at minimum).
- [ ] SGF export (standard Go file format, interoperable with every Go program).
- [ ] PNG export of the annotated board.

### Theming
- [ ] Board/stone themes — CSS custom properties + `<defs>` for gradients, shadows, wood textures, slate/shell stones.

### Infra / deploy
- [ ] CI: typecheck + test on PR.
- [ ] **Background ingest worker** — PDF rendering + board detection currently runs synchronously inside the upload request and is slow on Cloud Run's 2 vCPU. Move to Cloud Run Jobs or a Pub/Sub-driven worker so the user isn't blocked while a book ingests.
