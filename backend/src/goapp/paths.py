"""Central config for all data + model filesystem locations.

Two roots:

* **Models** — production weights (`*_detector.pt`). Live in the repo at
  `backend/data/models/` and are committed to git so a fresh clone is
  immediately runnable. Cloud Run mirrors this layout at `/app/data/models/`
  (baked in by the Dockerfile).
* **Data** — everything else: synth pages, derived YOLO datasets,
  validation sets, per-user libraries, transient uploads, training run
  artifacts. Lives under `$GOAPP_DATA_DIR` (default `~/data/go-app`,
  Cloud Run mounts the bucket at `/data`).

Override either with `GOAPP_MODELS_DIR` / `GOAPP_DATA_DIR`.

Subdirectory layout under $GOAPP_DATA_DIR:
    data/
        synth_pages/         synthetic page images + annotations
        synth_edge_crops/    derived: per-board crops with edge flags
        yolo/                derived: YOLO dataset built from synth_pages
        yolo_stones/         derived: YOLO stone-detector dataset
        bbox_test/           per-session PDF pages for the bbox tester
        tsumego/{user_id}/   per-user library of accepted problems
        uploads/{user_id}/   transient PDF uploads (signed-URL flow only)
        training_runs/       ultralytics training run artifacts
"""

from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    return Path(os.environ.get("GOAPP_DATA_DIR", Path.home() / "data" / "go-app"))


def _models_root() -> Path:
    if "GOAPP_MODELS_DIR" in os.environ:
        return Path(os.environ["GOAPP_MODELS_DIR"])
    # Repo-relative: backend/data/models/. Mirrored at /app/data/models/
    # in the Cloud Run image. Computed from this file's location so it
    # works whether the package is installed editable or copied into a
    # container at /app/src/goapp/.
    return Path(__file__).resolve().parents[2] / "data" / "models"


DATA_DIR = _data_root() / "data"
MODELS_DIR = _models_root()

# --- synth data (regenerable via goapp.synth) ---
SYNTH_PAGES_DIR = DATA_DIR / "synth_pages"

# --- YOLO derived dataset (built from synth pages) ---
YOLO_DIR = DATA_DIR / "yolo"

# --- per-session bbox-test data ---
BBOX_TEST_DIR = DATA_DIR / "bbox_test"

# --- accepted problems, saved from the upload flow (per-user) ---
TSUMEGO_ROOT = DATA_DIR / "tsumego"


def tsumego_dir(user_id: str) -> Path:
    return TSUMEGO_ROOT / user_id


# --- transient PDF uploads (signed-URL flow; deleted after ingest) ---
UPLOADS_ROOT = DATA_DIR / "uploads"


def uploads_dir(user_id: str) -> Path:
    return UPLOADS_ROOT / user_id


def uploads_object_key(user_id: str, upload_id: str) -> str:
    """GCS object key (relative to the bucket) for an upload.

    Mirrors the filesystem layout under DATA_DIR so the FUSE mount sees the
    same file the signed URL targets. The leading 'data/' segment matches
    DATA_DIR's position under the mount root (GOAPP_DATA_DIR=/data).
    """
    return f"data/uploads/{user_id}/{upload_id}.pdf"


# --- model weights ---
BOARD_DETECTOR_PATH = MODELS_DIR / "board_detector.pt"
STONE_DETECTOR_PATH = MODELS_DIR / "stone_detector.pt"
INTERSECTION_DETECTOR_PATH = MODELS_DIR / "intersection_detector.pt"
INTERSECTION_DETECTOR_NO_EDGES_PATH = MODELS_DIR / "intersection_detector_no_edges.pt"

# --- training run artifacts (ultralytics' project dir) ---
TRAINING_RUNS_DIR = DATA_DIR / "training_runs"
