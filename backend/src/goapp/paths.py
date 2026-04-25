"""Central config for all data + model filesystem locations.

Everything lives under $GOAPP_DATA_DIR. In production (Cloud Run) this is
a Cloud Storage FUSE mount; locally it defaults to ~/data/go-app. Model
weights are baked into the serving image at $GOAPP_MODELS_DIR (default
the repo's data/models, which is what `make deploy` copies in).

Subdirectory layout under $GOAPP_DATA_DIR:
    data/
        synth_pages/         synthetic page images + annotations
        synth_edge_crops/    derived: per-board crops with edge flags
        synth_grid_crops/    derived: per-board crops with 19x19 grid labels
        yolo/                derived: YOLO dataset built from synth_pages
        yolo_stones/         derived: YOLO stone-detector dataset
        bbox_test/           per-session PDF pages for the bbox tester
        tsumego/{user_id}/   per-user library of accepted problems
        uploads/{user_id}/   transient PDF uploads (signed-URL flow only)
    models/
        runs/                ultralytics training run artifacts
"""

from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    return Path(os.environ.get("GOAPP_DATA_DIR", Path.home() / "data" / "go-app"))


def _models_root() -> Path:
    if "GOAPP_MODELS_DIR" in os.environ:
        return Path(os.environ["GOAPP_MODELS_DIR"])
    return _data_root() / "models"


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

MODELS_RUNS_DIR = MODELS_DIR / "runs"
