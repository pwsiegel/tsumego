"""Train the snap-to-grid regressor on synth jittered tight crops.

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_snap \\
        --limit 3000 --epochs 12 --batch-size 32
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
from torch.utils.data import DataLoader, Dataset

from .snap_classifier import IMG_SIZE, SnapToGrid


from .paths import SNAP_CLASSIFIER_PATH as DEFAULT_MODEL
from .paths import SYNTH_SNAP_CROPS_DIR as DEFAULT_DATA
SEED = 42


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path


class SnapDataset(Dataset):
    def __init__(self, examples: list[Example], augment: bool = False):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = cv2.imread(str(ex.image_path), cv2.IMREAD_GRAYSCALE)
        label = json.loads(ex.label_path.read_text())
        if self.augment:
            alpha = 0.9 + random.random() * 0.2
            beta = (random.random() - 0.5) * 20
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        target = torch.tensor(
            [label["pitch_x"], label["pitch_y"], label["origin_x"], label["origin_y"]],
            dtype=torch.float32,
        )
        return tensor, target


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    examples: list[Example] = []
    for json_path in sorted(args.data_dir.glob("*.json")):
        img_path = json_path.with_suffix(".png")
        if img_path.exists():
            examples.append(Example(image_path=img_path, label_path=json_path))
    if not examples:
        raise RuntimeError(f"no snap crops in {args.data_dir}")

    random.seed(SEED)
    random.shuffle(examples)
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]
    split = max(1, int(len(examples) * (1 - args.val_frac)))
    train_ex, val_ex = examples[:split], examples[split:]
    print(f"examples: {len(examples)} → {len(train_ex)} train / {len(val_ex)} val", flush=True)

    device = pick_device()
    print(f"device: {device}", flush=True)

    train_loader = DataLoader(
        SnapDataset(train_ex, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        SnapDataset(val_ex, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = SnapToGrid().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tsum, tn = 0.0, 0
        for img, target in train_loader:
            img = img.to(device); target = target.to(device)
            pred = model(img)
            loss = loss_fn(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tsum += loss.item() * img.size(0); tn += img.size(0)
        tloss = tsum / max(1, tn)

        model.eval()
        vsum, vn = 0.0, 0
        # Track L1 error in pitch / origin separately (in fraction units)
        pitch_err_sum = 0.0
        origin_err_sum = 0.0
        with torch.no_grad():
            for img, target in val_loader:
                img = img.to(device); target = target.to(device)
                pred = model(img)
                vsum += loss_fn(pred, target).item() * img.size(0)
                vn += img.size(0)
                diff = (pred - target).abs()
                pitch_err_sum += diff[:, :2].mean(dim=1).sum().item()
                origin_err_sum += diff[:, 2:].mean(dim=1).sum().item()
        vloss = vsum / max(1, vn)
        p_err = pitch_err_sum / max(1, vn)
        o_err = origin_err_sum / max(1, vn)

        marker = ""
        if vloss < best_val:
            best_val = vloss
            epochs_since_best = 0
            torch.save({"model": model.state_dict(), "val_loss": vloss}, args.model_path)
            marker = "  [best → saved]"
        else:
            epochs_since_best += 1
        print(
            f"epoch {epoch:>3d}  train {tloss:.5f}  val {vloss:.5f}  "
            f"pitch_err {p_err:.4f}  origin_err {o_err:.4f}{marker}",
            flush=True,
        )
        if epochs_since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs", flush=True)
            break

    print(f"best val loss: {best_val:.5f} → {args.model_path}", flush=True)


if __name__ == "__main__":
    main()
