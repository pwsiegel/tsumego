"""Export the trained YOLO weights to ONNX for the lean serving image.

Run via `make export-models` (uses the [ml] extras for ultralytics).
Produces board_detector.onnx + stone_detector.onnx next to the .pt files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from goapp.paths import BOARD_DETECTOR_PATH, STONE_DETECTOR_PATH

IMG_SIZE = 640
OPSET = 12


def export(pt_path: Path) -> Path:
    if not pt_path.exists():
        raise FileNotFoundError(f"missing weights: {pt_path}")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    out = model.export(format="onnx", imgsz=IMG_SIZE, opset=OPSET, dynamic=False, simplify=True)
    onnx_path = Path(out) if isinstance(out, str) else pt_path.with_suffix(".onnx")
    print(f"exported {pt_path.name} -> {onnx_path}")
    return onnx_path


def main() -> int:
    for p in (BOARD_DETECTOR_PATH, STONE_DETECTOR_PATH):
        export(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
