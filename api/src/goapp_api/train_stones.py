"""Train a small UNet to predict stone-center heatmaps from board crops.

Each labeled example is (crop.png, {"black": [[x,y], ...], "white": [...]}).
The model outputs a 2-channel heatmap at the same spatial resolution as its
input: channel 0 peaks at black-stone centers, channel 1 at white-stone
centers. Loss is MSE against Gaussian-smoothed ground truth.

Usage:
    uv run --extra ml python -m goapp_api.train_stones
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


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "training_data" / "stone_points"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "stone_detector.pt"

IMG_SIZE = 512
HEATMAP_SIGMA = 8.0
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
VAL_FRAC = 0.2
SEED = 42
PATIENCE = 8


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path


def gather_examples() -> list[Example]:
    examples: list[Example] = []
    for json_path in sorted(DATA_DIR.glob("*.json")):
        img_path = json_path.with_suffix(".png")
        if img_path.exists():
            examples.append(Example(image_path=img_path, label_path=json_path))
    return examples


class StoneDataset(Dataset):
    def __init__(self, examples: list[Example], augment: bool = False):
        self.examples = examples
        self.augment = augment
        yy, xx = np.meshgrid(
            np.arange(IMG_SIZE, dtype=np.float32),
            np.arange(IMG_SIZE, dtype=np.float32),
            indexing="ij",
        )
        self._yy = yy
        self._xx = xx

    def __len__(self) -> int:
        return len(self.examples)

    def _gaussian(self, cx: float, cy: float) -> np.ndarray:
        d2 = (self._xx - cx) ** 2 + (self._yy - cy) ** 2
        return np.exp(-d2 / (2 * HEATMAP_SIGMA * HEATMAP_SIGMA))

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = cv2.imread(str(ex.image_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        data = json.loads(ex.label_path.read_text())
        blacks = data.get("black", [])
        whites = data.get("white", [])

        img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        sx, sy = IMG_SIZE / w, IMG_SIZE / h
        blacks_s = [(x * sx, y * sy) for x, y in blacks]
        whites_s = [(x * sx, y * sy) for x, y in whites]

        if self.augment:
            if random.random() < 0.5:
                img_r = np.ascontiguousarray(img_r[:, ::-1])
                blacks_s = [(IMG_SIZE - 1 - x, y) for x, y in blacks_s]
                whites_s = [(IMG_SIZE - 1 - x, y) for x, y in whites_s]
            if random.random() < 0.5:
                img_r = np.ascontiguousarray(img_r[::-1, :])
                blacks_s = [(x, IMG_SIZE - 1 - y) for x, y in blacks_s]
                whites_s = [(x, IMG_SIZE - 1 - y) for x, y in whites_s]
            rot = random.randint(0, 3)
            for _ in range(rot):
                img_r = np.ascontiguousarray(np.rot90(img_r))
                blacks_s = [(y, IMG_SIZE - 1 - x) for x, y in blacks_s]
                whites_s = [(y, IMG_SIZE - 1 - x) for x, y in whites_s]
            # Mild brightness/contrast jitter so the model doesn't over-fit to
            # any single book's exposure.
            alpha = 0.9 + random.random() * 0.2
            beta = (random.random() - 0.5) * 20
            img_r = np.clip(alpha * img_r.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img_r).float().unsqueeze(0) / 255.0

        heatmap = np.zeros((2, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for x, y in blacks_s:
            np.maximum(heatmap[0], self._gaussian(x, y), out=heatmap[0])
        for x, y in whites_s:
            np.maximum(heatmap[1], self._gaussian(x, y), out=heatmap[1])
        return img_t, torch.from_numpy(heatmap)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore[override]
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.u3 = DoubleConv(base * (8 + 4), base * 4)
        self.u2 = DoubleConv(base * (4 + 2), base * 2)
        self.u1 = DoubleConv(base * (2 + 1), base)
        self.out = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):  # type: ignore[override]
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        u3 = self.u3(torch.cat([self.up(d4), d3], dim=1))
        u2 = self.u2(torch.cat([self.up(u3), d2], dim=1))
        u1 = self.u1(torch.cat([self.up(u2), d1], dim=1))
        return torch.sigmoid(self.out(u1))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="directory containing <id>.png/<id>.json pairs")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH,
                    help="where to save the trained model")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--limit", type=int, default=None,
                    help="randomly subsample this many examples before training")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes (0 = main thread)")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    model_path: Path = args.model_path

    examples: list[Example] = []
    for json_path in sorted(data_dir.glob("*.json")):
        img_path = json_path.with_suffix(".png")
        if img_path.exists():
            examples.append(Example(image_path=img_path, label_path=json_path))
    if not examples:
        raise RuntimeError(f"no stone-point labels found in {data_dir}")

    random.seed(SEED)
    random.shuffle(examples)
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]
    split = max(1, int(len(examples) * (1 - args.val_frac)))
    train_ex, val_ex = examples[:split], examples[split:]
    print(f"examples: {len(examples)} total → {len(train_ex)} train / {len(val_ex)} val")
    print(f"data: {data_dir}")
    print(f"model out: {model_path}")

    device = pick_device()
    print(f"device: {device}")

    train_loader = DataLoader(
        StoneDataset(train_ex, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        StoneDataset(val_ex, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        for img, target in train_loader:
            img = img.to(device); target = target.to(device)
            pred = model(img)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * img.size(0)
            train_n += img.size(0)
        train_loss = train_sum / max(1, train_n)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for img, target in val_loader:
                img = img.to(device); target = target.to(device)
                val_sum += loss_fn(model(img), target).item() * img.size(0)
                val_n += img.size(0)
        val_loss = val_sum / max(1, val_n)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_since_best = 0
            torch.save({"model": model.state_dict(), "val_loss": val_loss}, model_path)
            marker = "  [best → saved]"
        else:
            epochs_since_best += 1

        print(f"epoch {epoch:>3d}  train {train_loss:.6f}  val {val_loss:.6f}{marker}")

        if epochs_since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs")
            break

    print(f"best val loss: {best_val:.6f} → saved to {model_path}")


if __name__ == "__main__":
    main()
