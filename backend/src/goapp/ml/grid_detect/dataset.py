"""Dataset of (board crop → grid geometry) pairs derived from synth pages.

Crops are extracted on-the-fly from the existing ``synth_pages/`` corpus.
Each item:

  Input  — RGB tensor, ImageNet-normalized, ``IMG_SIZE × IMG_SIZE``.
  Label  — 6 floats in [0, 1]: (gx1, gy1, gx2, gy2, px, py), all
           expressed as a fraction of the input crop's width/height.

Augmentations during training:

  - Per-side jitter on the crop bounds (mimics YOLO bbox slop at inference).
  - Small in-plane rotation + shear (mimics photocopier tilt and warp).
  - Mild brightness/contrast jitter.

The synth label already contains everything we need: ``board["bbox"]`` is
the tight grid-intersection rectangle in page coordinates, and
``board["window"]`` gives ``n_cols`` / ``n_rows`` so we can derive pitch.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMG_SIZE = 224

JITTER_MIN_PX = 0
JITTER_MAX_PX = 25

WARP_PROB = 0.6
WARP_MAX_DEG = 3.0          # rotation, mimics scan tilt
WARP_MAX_SHEAR_DEG = 2.5    # mild planar shear, mimics photocopy warp

COLOR_PROB = 0.7

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GridCropsDataset(Dataset):
    def __init__(
        self,
        page_jsons: list[Path],
        img_size: int = IMG_SIZE,
        augment: bool = True,
    ) -> None:
        self.img_size = img_size
        self.augment = augment

        self.items: list[tuple[Path, dict]] = []
        for jp in page_jsons:
            try:
                data = json.loads(jp.read_text())
            except Exception:
                continue
            png = jp.with_suffix(".png")
            if not png.exists():
                continue
            for board in data.get("boards", []):
                if "bbox" in board and "window" in board:
                    self.items.append((png, board))

        self._page_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load_page(self, png_path: Path) -> np.ndarray | None:
        cached = self._page_cache.get(png_path)
        if cached is not None:
            return cached
        arr = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            self._page_cache[png_path] = img
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        png_path, board = self.items[idx]
        page = self._load_page(png_path)
        if page is None:
            return self.__getitem__((idx + 1) % len(self))
        Hp, Wp = page.shape[:2]

        gx0, gy0, gx1, gy1 = board["bbox"]
        col_min, col_max, row_min, row_max = board["window"]
        n_cols = max(1, col_max - col_min)
        n_rows = max(1, row_max - row_min)

        if self.augment:
            jl = random.randint(JITTER_MIN_PX, JITTER_MAX_PX)
            jt = random.randint(JITTER_MIN_PX, JITTER_MAX_PX)
            jr = random.randint(JITTER_MIN_PX, JITTER_MAX_PX)
            jb = random.randint(JITTER_MIN_PX, JITTER_MAX_PX)
        else:
            jl = jt = jr = jb = (JITTER_MIN_PX + JITTER_MAX_PX) // 2

        px0, py0, px1, py1 = board.get("bbox_padded", board["bbox"])
        cx0 = max(0, int(px0 - jl))
        cy0 = max(0, int(py0 - jt))
        cx1 = min(Wp, int(px1 + jr) + 1)
        cy1 = min(Hp, int(py1 + jb) + 1)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return self.__getitem__((idx + 1) % len(self))

        crop = page[cy0:cy1, cx0:cx1].copy()

        local_gx0 = float(gx0 - cx0)
        local_gy0 = float(gy0 - cy0)
        local_gx1 = float(gx1 - cx0)
        local_gy1 = float(gy1 - cy0)
        pitch_x_px = (gx1 - gx0) / n_cols
        pitch_y_px = (gy1 - gy0) / n_rows

        if self.augment and random.random() < WARP_PROB:
            crop, (local_gx0, local_gy0, local_gx1, local_gy1,
                   pitch_x_px, pitch_y_px) = _apply_warp(
                crop,
                local_gx0, local_gy0, local_gx1, local_gy1,
                n_cols, n_rows,
            )

        Ch, Cw = crop.shape[:2]

        if self.augment and random.random() < COLOR_PROB:
            crop = _color_jitter(crop)

        resized = cv2.resize(
            crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA,
        )

        label = torch.tensor(
            [
                local_gx0 / Cw,
                local_gy0 / Ch,
                local_gx1 / Cw,
                local_gy1 / Ch,
                pitch_x_px / Cw,
                pitch_y_px / Ch,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        return tensor, label


def _apply_warp(
    crop: np.ndarray,
    gx0: float, gy0: float, gx1: float, gy1: float,
    n_cols: int, n_rows: int,
):
    """Apply a small rotation + shear to the crop and re-derive label.

    The label is recomputed by transforming the four grid corners and
    taking the axis-aligned bbox of the result. Pitch is the bbox extent
    divided by ``n_cols`` / ``n_rows``. For small warp angles this
    approximation matches the local pitch closely.
    """
    Ch, Cw = crop.shape[:2]
    deg = random.uniform(-WARP_MAX_DEG, WARP_MAX_DEG)
    shx = random.uniform(-WARP_MAX_SHEAR_DEG, WARP_MAX_SHEAR_DEG)
    shy = random.uniform(-WARP_MAX_SHEAR_DEG, WARP_MAX_SHEAR_DEG)

    cx_, cy_ = Cw / 2.0, Ch / 2.0
    R = np.vstack([cv2.getRotationMatrix2D((cx_, cy_), deg, 1.0), [0, 0, 1]])
    tx = math.tan(math.radians(shx))
    ty = math.tan(math.radians(shy))
    S = np.array([[1, tx, 0], [ty, 1, 0], [0, 0, 1]], dtype=np.float64)
    T0 = np.array([[1, 0, -cx_], [0, 1, -cy_], [0, 0, 1]], dtype=np.float64)
    T1 = np.array([[1, 0, cx_], [0, 1, cy_], [0, 0, 1]], dtype=np.float64)
    M3 = R @ T1 @ S @ T0
    M = M3[:2, :].astype(np.float32)

    warped = cv2.warpAffine(
        crop, M, (Cw, Ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    corners = np.array(
        [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]],
        dtype=np.float32,
    )
    ones = np.ones((4, 1), dtype=np.float32)
    h_corners = np.hstack([corners, ones])
    new_corners = h_corners @ M.T  # (4, 2)
    nx0, nx1 = float(new_corners[:, 0].min()), float(new_corners[:, 0].max())
    ny0, ny1 = float(new_corners[:, 1].min()), float(new_corners[:, 1].max())
    new_pitch_x = (nx1 - nx0) / n_cols
    new_pitch_y = (ny1 - ny0) / n_rows
    return warped, (nx0, ny0, nx1, ny1, new_pitch_x, new_pitch_y)


def _color_jitter(img: np.ndarray) -> np.ndarray:
    alpha = 1.0 + random.uniform(-0.15, 0.15)
    beta = random.uniform(-15, 15)
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)
