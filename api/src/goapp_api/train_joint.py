"""Joint fine-tune of stone CNN + snap regressor on grid-level loss.

Approach: for each training crop, run the stone CNN (heatmap) and the snap
regressor (pitch/origin). Compute predicted grid-intersection pixel
positions from snap output, bilinearly sample the heatmap at those
positions, and treat the sampled values as per-intersection B/W activations.
Cross-entropy against a ground-truth 19x19 grid (with optional Gaussian
smoothing over nearby cells).

Auxiliary losses:
  - snap aux: smooth-L1 on pitch/origin vs GT (keeps geometry precise even
    when grid CE is locally small)

Pre-trained stone CNN + snap weights are loaded at start; this script just
fine-tunes them jointly.

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_joint \\
        --epochs 8 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .snap_classifier import IMG_SIZE as SNAP_SIZE, SnapToGrid
from .train_stones import IMG_SIZE as STONE_SIZE, UNet


from .paths import (
    SNAP_CLASSIFIER_JOINT_PATH as JOINT_SNAP_OUT,
    SNAP_CLASSIFIER_PATH as SNAP_WEIGHTS,
    STONE_DETECTOR_JOINT_PATH as JOINT_STONE_OUT,
    STONE_DETECTOR_PATH as STONE_WEIGHTS,
    SYNTH_GRID_CROPS_DIR as DEFAULT_DATA,
)
SEED = 42
BOARD_SIZE = 19


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path


class GridDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = cv2.imread(str(ex.image_path), cv2.IMREAD_GRAYSCALE)
        label = json.loads(ex.label_path.read_text())
        grid = np.array(label["grid"], dtype=np.int64)   # (19,19)
        mask = np.array(label["mask"], dtype=np.float32)  # (19,19)
        window = label["window"]
        col_min, col_max, row_min, row_max = window

        # Two resized views: snap uses 256, stone CNN uses 512.
        snap_in = cv2.resize(img, (SNAP_SIZE, SNAP_SIZE), interpolation=cv2.INTER_AREA)
        stone_in = cv2.resize(img, (STONE_SIZE, STONE_SIZE), interpolation=cv2.INTER_AREA)

        snap_t = torch.from_numpy(snap_in).float().unsqueeze(0) / 255.0
        stone_t = torch.from_numpy(stone_in).float().unsqueeze(0) / 255.0

        # GT snap params (tight crop → origin=0, pitch = 1/(n-1)).
        n_cols = max(1, col_max - col_min)
        n_rows = max(1, row_max - row_min)
        snap_gt = torch.tensor([
            1.0 / n_cols, 1.0 / n_rows, 0.0, 0.0,
        ], dtype=torch.float32)

        return (
            stone_t, snap_t, torch.from_numpy(grid), torch.from_numpy(mask),
            torch.tensor(window, dtype=torch.int64), snap_gt,
        )


def sample_grid_from_heatmap(
    heatmap: torch.Tensor,           # (B, 2, H, W)
    snap_out: torch.Tensor,          # (B, 4): pitch_x, pitch_y, origin_x, origin_y (fractions)
    window: torch.Tensor,            # (B, 4): col_min, col_max, row_min, row_max
) -> torch.Tensor:
    """Bilinearly sample the heatmap at each 19x19 intersection's predicted
    pixel position. Returns (B, 2, 19, 19) sampled values."""
    B = heatmap.size(0)
    device = heatmap.device
    col_min = window[:, 0].float(); col_max = window[:, 1].float()
    row_min = window[:, 2].float(); row_max = window[:, 3].float()

    # For intersections inside the visible window, compute their pixel
    # positions (as fractions of crop dim) via snap output.
    col_idx = torch.arange(BOARD_SIZE, device=device, dtype=torch.float32)  # (19,)
    row_idx = torch.arange(BOARD_SIZE, device=device, dtype=torch.float32)  # (19,)

    pitch_x = snap_out[:, 0]  # (B,)
    pitch_y = snap_out[:, 1]
    origin_x = snap_out[:, 2]
    origin_y = snap_out[:, 3]

    # Position of intersection (c, r) as fraction of width:
    #   x_frac = origin_x + (c - col_min) * pitch_x
    #   y_frac = origin_y + (r - row_min) * pitch_y
    # where col_min is the leftmost visible column.
    # We produce (B, 19, 19) position tensors.
    col_delta = col_idx[None, :] - col_min[:, None]  # (B, 19)
    row_delta = row_idx[None, :] - row_min[:, None]  # (B, 19)
    xs = origin_x[:, None] + col_delta * pitch_x[:, None]      # (B, 19)
    ys = origin_y[:, None] + row_delta * pitch_y[:, None]      # (B, 19)

    # Broadcast to (B, 19, 19): y is row, x is col
    grid_x = xs[:, None, :].expand(B, BOARD_SIZE, BOARD_SIZE)  # cols vary along last dim
    grid_y = ys[:, :, None].expand(B, BOARD_SIZE, BOARD_SIZE)  # rows vary along middle dim

    # grid_sample expects coords in [-1, 1]; our fractions are in [0, 1].
    # Also expects (B, H_out, W_out, 2) — last dim is (x, y).
    sample_x = grid_x * 2 - 1
    sample_y = grid_y * 2 - 1
    sample_grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, 19, 19, 2)

    sampled = F.grid_sample(
        heatmap, sample_grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )  # (B, 2, 19, 19)
    return sampled


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--snap-weight", type=float, default=1.0)
    args = ap.parse_args()

    examples: list[Example] = []
    for json_path in sorted(args.data_dir.glob("*.json")):
        img_path = json_path.with_suffix(".png")
        if img_path.exists():
            examples.append(Example(image_path=img_path, label_path=json_path))
    if not examples:
        raise RuntimeError(f"no grid crops in {args.data_dir}")

    random.seed(SEED); random.shuffle(examples)
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]
    split = max(1, int(len(examples) * (1 - args.val_frac)))
    train_ex, val_ex = examples[:split], examples[split:]
    print(f"examples: {len(examples)} → {len(train_ex)} train / {len(val_ex)} val", flush=True)

    device = pick_device()
    print(f"device: {device}", flush=True)

    # Load pre-trained weights.
    stone_model = UNet()
    stone_model.load_state_dict(torch.load(str(STONE_WEIGHTS), map_location="cpu")["model"])
    stone_model.to(device)

    snap_model = SnapToGrid()
    snap_model.load_state_dict(torch.load(str(SNAP_WEIGHTS), map_location="cpu")["model"])
    snap_model.to(device)

    params = list(stone_model.parameters()) + list(snap_model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    class_weights = torch.tensor([0.4, 1.0, 1.0], device=device)
    ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    snap_l1 = nn.SmoothL1Loss()

    train_loader = DataLoader(
        GridDataset(train_ex), batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        GridDataset(val_ex), batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    best_val = float("inf")
    epochs_since_best = 0

    def forward(stone_t, snap_t, grid, mask, window, snap_gt):
        heatmap = stone_model(stone_t)          # (B, 2, 512, 512); sigmoid-activated
        snap_out = snap_model(snap_t)           # (B, 4)
        sampled = sample_grid_from_heatmap(heatmap, snap_out, window)  # (B, 2, 19, 19)
        # Treat sampled values as per-cell B/W probabilities (stone CNN
        # output is already sigmoid). Empty prob = 1 - B - W (clamped).
        sampled = sampled.clamp(0.0, 1.0)
        prob_b = sampled[:, 0:1]
        prob_w = sampled[:, 1:2]
        prob_empty = (1.0 - prob_b - prob_w).clamp(min=1e-6)
        probs = torch.cat([prob_empty, prob_b, prob_w], dim=1)  # (B, 3, 19, 19)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        log_probs = torch.log(probs.clamp(min=1e-6))

        per_cell = F.nll_loss(log_probs, grid, weight=class_weights, reduction="none")
        grid_loss = (per_cell * mask).sum() / mask.sum().clamp(min=1.0)
        snap_loss = snap_l1(snap_out, snap_gt)
        total = grid_loss + args.snap_weight * snap_loss
        return total, grid_loss, snap_loss, log_probs

    for epoch in range(1, args.epochs + 1):
        stone_model.train(); snap_model.train()
        tsum, tn = 0.0, 0
        gsum = 0.0; ssum = 0.0
        for stone_t, snap_t, grid, mask, window, snap_gt in train_loader:
            stone_t = stone_t.to(device); snap_t = snap_t.to(device)
            grid = grid.to(device); mask = mask.to(device)
            window = window.to(device); snap_gt = snap_gt.to(device)
            total, gl, sl, _ = forward(stone_t, snap_t, grid, mask, window, snap_gt)
            opt.zero_grad(); total.backward(); opt.step()
            b = stone_t.size(0)
            tsum += total.item() * b; tn += b
            gsum += gl.item() * b; ssum += sl.item() * b
        tloss = tsum / max(1, tn)
        tgrid = gsum / max(1, tn); tsnap = ssum / max(1, tn)

        stone_model.eval(); snap_model.eval()
        vsum, vn = 0.0, 0
        correct_cells = 0; total_cells = 0
        with torch.no_grad():
            for stone_t, snap_t, grid, mask, window, snap_gt in val_loader:
                stone_t = stone_t.to(device); snap_t = snap_t.to(device)
                grid = grid.to(device); mask = mask.to(device)
                window = window.to(device); snap_gt = snap_gt.to(device)
                total, _, _, logits = forward(stone_t, snap_t, grid, mask, window, snap_gt)
                b = stone_t.size(0)
                vsum += total.item() * b; vn += b
                preds = logits.argmax(dim=1)
                m_bool = mask.bool()
                correct_cells += ((preds == grid) & m_bool).sum().item()
                total_cells += m_bool.sum().item()
        vloss = vsum / max(1, vn)
        cell_acc = correct_cells / max(1, total_cells)

        marker = ""
        if vloss < best_val:
            best_val = vloss
            epochs_since_best = 0
            torch.save({"model": stone_model.state_dict()}, JOINT_STONE_OUT)
            torch.save({"model": snap_model.state_dict()}, JOINT_SNAP_OUT)
            marker = "  [best → saved]"
        else:
            epochs_since_best += 1
        print(
            f"epoch {epoch:>3d}  train {tloss:.4f}  (grid {tgrid:.4f}  snap {tsnap:.4f})  "
            f"val {vloss:.4f}  cell_acc {cell_acc:.4f}{marker}",
            flush=True,
        )
        if epochs_since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs", flush=True)
            break

    print(f"best val loss: {best_val:.4f}  saved to {JOINT_STONE_OUT} and {JOINT_SNAP_OUT}", flush=True)


if __name__ == "__main__":
    main()
