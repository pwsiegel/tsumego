"""Central config for all data + model filesystem locations.

Everything lives under $GOAPP_DATA_DIR (default ~/data/go-app). No data or
weights live inside the repo, so `git clean -dfx` in the repo is safe.

Subdirectory layout under $GOAPP_DATA_DIR:
    data/
        synth_pages/         synthetic page images + annotations
        synth_edge_crops/    derived: per-board crops with edge flags
        synth_grid_crops/    derived: per-board crops with 19x19 grid labels
        yolo/                derived: YOLO dataset built from synth_pages
        yolo_stones/         derived: YOLO stone-detector dataset
        bbox_test/           per-session PDF pages for the bbox tester
    models/
        board_detector.pt    YOLO board bbox detector
        edge_classifier.pt   4-bit edge classifier
        stone_detector.pt    YOLO stone detector (classes: B, W)
        runs/                ultralytics training run artifacts
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("GOAPP_DATA_DIR", Path.home() / "data" / "go-app"))


DATA_DIR = _root() / "data"
MODELS_DIR = _root() / "models"

# --- synth data (regenerable via goapp_api.synth) ---
SYNTH_PAGES_DIR = DATA_DIR / "synth_pages"
SYNTH_EDGE_CROPS_DIR = DATA_DIR / "synth_edge_crops"
SYNTH_GRID_CROPS_DIR = DATA_DIR / "synth_grid_crops"

# --- YOLO derived dataset (built from synth pages) ---
YOLO_DIR = DATA_DIR / "yolo"

# --- per-session bbox-test data ---
BBOX_TEST_DIR = DATA_DIR / "bbox_test"

# --- model weights ---
BOARD_DETECTOR_PATH = MODELS_DIR / "board_detector.pt"
EDGE_CLASSIFIER_PATH = MODELS_DIR / "edge_classifier.pt"
STONE_DETECTOR_PATH = MODELS_DIR / "stone_detector.pt"

MODELS_RUNS_DIR = MODELS_DIR / "runs"
