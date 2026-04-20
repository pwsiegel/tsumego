"""Central config for all data + model filesystem locations.

Everything lives under $GOAPP_DATA_DIR (default ~/data/go-app). No data or
weights live inside the repo, so `git clean -dfx` in the repo is safe.

Subdirectory layout under $GOAPP_DATA_DIR:
    data/
        boards_deprecated/   hand-labeled board bboxes (deprecated, archival)
        stone_points/        hand-labeled stone-center crops (real data)
        stone_tasks/         per-ingest PDF crops (regenerated each ingest)
        synth_pages/         synthetic page images + annotations
        synth_stone_crops/   derived: per-board crops with stone centers
        synth_edge_crops/    derived: per-board crops with edge flags
        synth_grid_crops/    derived: per-board crops with 19x19 grid labels
        synth_snap_crops/    derived: jittered crops with pitch/origin labels
        yolo/                derived: real-data YOLO dataset (legacy)
        yolo_synth/          derived: synth-data YOLO dataset
    models/
        board_detector.pt    YOLO board bbox detector
        edge_classifier.pt   4-bit edge classifier
        stone_detector.pt    stone-center CNN (UNet)
        snap_classifier.pt   pitch/origin regressor
        grid_classifier.pt   19x19 grid classifier (experimental)
        *_joint.pt           joint-fine-tuned variants (experimental)
        runs/                ultralytics training run artifacts
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("GOAPP_DATA_DIR", Path.home() / "data" / "go-app"))


DATA_DIR = _root() / "data"
MODELS_DIR = _root() / "models"

# --- hand-labeled / durable data ---
BOARDS_DEPRECATED_DIR = DATA_DIR / "boards_deprecated"
STONE_POINTS_DIR = DATA_DIR / "stone_points"

# --- per-session ingest data ---
STONE_TASKS_DIR = DATA_DIR / "stone_tasks"

# --- synth data (regenerable via goapp_api.synth) ---
SYNTH_PAGES_DIR = DATA_DIR / "synth_pages"
SYNTH_STONE_CROPS_DIR = DATA_DIR / "synth_stone_crops"
SYNTH_EDGE_CROPS_DIR = DATA_DIR / "synth_edge_crops"
SYNTH_GRID_CROPS_DIR = DATA_DIR / "synth_grid_crops"
SYNTH_SNAP_CROPS_DIR = DATA_DIR / "synth_snap_crops"

# --- YOLO derived datasets ---
YOLO_REAL_DIR = DATA_DIR / "yolo"
YOLO_SYNTH_DIR = DATA_DIR / "yolo_synth"

# --- model weights ---
BOARD_DETECTOR_PATH = MODELS_DIR / "board_detector.pt"
EDGE_CLASSIFIER_PATH = MODELS_DIR / "edge_classifier.pt"
STONE_DETECTOR_PATH = MODELS_DIR / "stone_detector.pt"
SNAP_CLASSIFIER_PATH = MODELS_DIR / "snap_classifier.pt"
GRID_CLASSIFIER_PATH = MODELS_DIR / "grid_classifier.pt"
STONE_DETECTOR_JOINT_PATH = MODELS_DIR / "stone_detector_joint.pt"
SNAP_CLASSIFIER_JOINT_PATH = MODELS_DIR / "snap_classifier_joint.pt"

MODELS_RUNS_DIR = MODELS_DIR / "runs"
