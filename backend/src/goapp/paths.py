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
        attempts/{user_id}/
            attempt_*.json   per-attempt records (sent_to / reviews embedded)
            teachers/{teacher_id}.json   one file per teacher (token + label)
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


# --- ingest jobs: per-(user, job_id) state + staged source PDF ---
INGEST_JOBS_ROOT = DATA_DIR / "ingest_jobs"


def ingest_jobs_dir(user_id: str) -> Path:
    return INGEST_JOBS_ROOT / user_id


def ingest_job_dir(user_id: str, job_id: str) -> Path:
    return ingest_jobs_dir(user_id) / job_id


# --- attempts: per-user solve attempts + teacher reviews ---
ATTEMPTS_ROOT = DATA_DIR / "attempts"


def attempts_dir(user_id: str) -> Path:
    return ATTEMPTS_ROOT / user_id


# --- teachers: per-(user, teacher) capability-URL files ---
def teachers_dir(user_id: str) -> Path:
    return attempts_dir(user_id) / "teachers"


def teacher_path(user_id: str, teacher_id: str) -> Path:
    return teachers_dir(user_id) / f"{teacher_id}.json"


# --- model weights ---
# .pt files are the training-format weights (used by ultralytics/torch in
# train + export pipelines). .onnx files are the serving-format weights
# loaded by onnxruntime in the lean Cloud Run image. Export via
# `make export-models`.
BOARD_DETECTOR_PATH = MODELS_DIR / "board_detector.pt"
STONE_DETECTOR_PATH = MODELS_DIR / "stone_detector.pt"
BOARD_DETECTOR_ONNX = MODELS_DIR / "board_detector.onnx"
STONE_DETECTOR_ONNX = MODELS_DIR / "stone_detector.onnx"

# --- training run artifacts (ultralytics' project dir) ---
TRAINING_RUNS_DIR = DATA_DIR / "training_runs"
