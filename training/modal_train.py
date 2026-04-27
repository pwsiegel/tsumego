"""Modal app for training the board + stone detectors on L4 GPUs.

Architecture:
  - Volume `tsumego-data` mounted at /vol holds inputs (synth_pages/) and
    outputs (models/, training_runs/).
  - Functions run on L4 GPUs; the image bundles ultralytics + headless cv2.
  - Source ships from backend/src/goapp into /app/goapp; PYTHONPATH=/app.
  - paths.py reads GOAPP_DATA_DIR=/vol so DATA_DIR=/vol/data and synth_pages
    sit at /vol/data/synth_pages. GOAPP_MODELS_DIR=/vol/models keeps trained
    weights on the volume so `modal volume get` can pull them locally.

Usage:
    # one-time: upload local synth_pages/ to the volume
    make modal-upload-synth      # or: modal volume put tsumego-data <local> /data/synth_pages

    # smoke test (~couple minutes, 50 pages, 2 epochs)
    make modal-train-smoke

    # full runs
    make modal-train-boards
    make modal-train-stones

    # pull trained weights into backend/data/models/
    make modal-pull-weights
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
GOAPP_SRC = REPO_ROOT / "backend" / "src" / "goapp"

VOLUME_NAME = "tsumego-data"
VOLUME_MOUNT = "/vol"

GPU_KIND = "L4"
TRAIN_TIMEOUT = 60 * 60 * 4   # 4h cap on full runs
SMOKE_TIMEOUT = 60 * 30       # 30m cap on smoke

app = modal.App("tsumego-training")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1", "libglib2.0-0",
        # Fonts the synth generator's text_sources.py expects on Linux:
        # noto-cjk covers ko/ja/zh; liberation provides the Times/Helvetica
        # equivalents used for Latin scripts.
        "fonts-noto-cjk", "fonts-liberation",
    )
    .pip_install(
        "ultralytics>=8.3",
        "torch>=2.5",
        "torchvision>=0.20",
        "Pillow>=11.0",
        "numpy>=2.1",
    )
    # ultralytics pulls opencv-python (GUI wheel); force the headless variant
    # so cv2 imports work in a containerised env without X11.
    .run_commands(
        'pip install --force-reinstall --no-deps "opencv-python-headless>=4.10"'
    )
    .env({
        "GOAPP_DATA_DIR": VOLUME_MOUNT,
        "GOAPP_MODELS_DIR": f"{VOLUME_MOUNT}/models",
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(str(GOAPP_SRC), remote_path="/app/goapp")
)


def _run(*args: str) -> None:
    """Run a goapp training module and surface its output live."""
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: data_volume},
    cpu=16,
    timeout=60 * 30,
)
def gen_synth(count: int = 1500) -> None:
    """Wipe and regenerate the synth_pages dataset on the volume."""
    import shutil
    pages_dir = Path(f"{VOLUME_MOUNT}/data/synth_pages")
    if pages_dir.exists():
        print(f"removing existing {pages_dir} ({sum(1 for _ in pages_dir.iterdir())} entries)")
        shutil.rmtree(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    _run("python", "-m", "goapp.synth.gen", "--count", str(count))
    data_volume.commit()


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=SMOKE_TIMEOUT,
)
def smoke(limit: int = 50, epochs: int = 2) -> None:
    """End-to-end smoke: tiny board-detector run that exercises the full path."""
    _run(
        "python", "-m", "goapp.ml.board_detect.train",
        "--limit", str(limit),
        "--epochs", str(epochs),
        "--model-out", f"{VOLUME_MOUNT}/models/_smoke_board_detector.pt",
    )
    data_volume.commit()


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=TRAIN_TIMEOUT,
)
def train_boards(limit: int = 500, epochs: int = 25) -> None:
    _run(
        "python", "-m", "goapp.ml.board_detect.train",
        "--limit", str(limit),
        "--epochs", str(epochs),
    )
    data_volume.commit()


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=TRAIN_TIMEOUT,
)
def train_stones(limit: int = 1500, epochs: int = 30) -> None:
    # train.py defaults --model-out to stone_detector_yolo.pt; pass explicit
    # path so the artifact lands where modal-pull-weights expects it.
    _run(
        "python", "-m", "goapp.ml.stone_detect.train",
        "--limit", str(limit),
        "--epochs", str(epochs),
        "--device", "0",
        "--model-out", f"{VOLUME_MOUNT}/models/stone_detector.pt",
    )
    data_volume.commit()


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=TRAIN_TIMEOUT,
)
def train_intersections(limit: int = 1500, epochs: int = 30) -> None:
    _run(
        "python", "-m", "goapp.ml.intersection_detect.train",
        "--limit", str(limit),
        "--epochs", str(epochs),
        "--device", "0",
        "--model-out", f"{VOLUME_MOUNT}/models/intersection_detector.pt",
    )
    data_volume.commit()


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=TRAIN_TIMEOUT,
)
def train_intersections_no_edges(limit: int = 1500, epochs: int = 30) -> None:
    """Diagnostic run: drop board-edge T/L labels so the model only ever sees
    "+" intersections. If predictions still hallucinate near the bbox boundary,
    the model is learning bbox-relative layout, not the intersection shape."""
    _run(
        "python", "-m", "goapp.ml.intersection_detect.train",
        "--limit", str(limit),
        "--epochs", str(epochs),
        "--device", "0",
        "--no-edge-labels",
        "--model-out", f"{VOLUME_MOUNT}/models/intersection_detector_no_edges.pt",
    )
    data_volume.commit()


