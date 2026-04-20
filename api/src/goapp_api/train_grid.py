"""Train the 19x19 grid classifier on synth crops.

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_grid \\
        --data-dir <path> --model-path <path> --epochs 10 --batch-size 16
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

from .grid_classifier import BOARD_SIZE, IMG_SIZE, GridClassifier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "training_data" / "synth_grid_crops"
DEFAULT_MODEL = ROOT / "models" / "grid_classifier.pt"
SEED = 42


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path


class GridDataset(Dataset):
    def __init__(self, examples: list[Example], augment: bool = False):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = cv2.imread(str(ex.image_path), cv2.IMREAD_GRAYSCALE)
        label = json.loads(ex.label_path.read_text())
        grid = np.array(label["grid"], dtype=np.int64)    # (19,19)
        mask = np.array(label["mask"], dtype=np.float32)  # (19,19)

        if self.augment:
            # Only brightness/contrast jitter — spatial jitter misaligns
            # stones with their grid-cell labels and wrecks training.
            alpha = 0.9 + random.random() * 0.2
            beta = (random.random() - 0.5) * 20
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor, torch.from_numpy(grid), torch.from_numpy(mask)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gather(data_dir: Path) -> list[Example]:
    out: list[Example] = []
    for json_path in sorted(data_dir.glob("*.json")):
        img_path = json_path.with_suffix(".png")
        if img_path.exists():
            out.append(Example(image_path=img_path, label_path=json_path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    examples = gather(args.data_dir)
    if not examples:
        raise RuntimeError(f"no grid labels in {args.data_dir}")

    random.seed(SEED)
    random.shuffle(examples)
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]
    split = max(1, int(len(examples) * (1 - args.val_frac)))
    train_ex, val_ex = examples[:split], examples[split:]
    print(f"examples: {len(examples)} total → {len(train_ex)} train / {len(val_ex)} val", flush=True)
    print(f"data: {args.data_dir}", flush=True)
    print(f"model out: {args.model_path}", flush=True)

    device = pick_device()
    print(f"device: {device}", flush=True)

    train_loader = DataLoader(
        GridDataset(train_ex, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        GridDataset(val_ex, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = GridClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Weighted CE: empty cells dominate (~90% of visible-mask cells), so
    # down-weight them moderately. 0.4 worked better than 0.1 in earlier
    # runs (0.1 caused the model to over-predict stones).
    class_weights = torch.tensor([0.4, 1.0, 1.0], device=device)
    ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tsum, tn = 0.0, 0
        for img, grid, mask in train_loader:
            img = img.to(device); grid = grid.to(device); mask = mask.to(device)
            logits = model(img)  # (B, 3, 19, 19)
            per_cell = ce(logits, grid)  # (B, 19, 19)
            loss = (per_cell * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tsum += loss.item() * img.size(0); tn += img.size(0)
        tloss = tsum / max(1, tn)

        model.eval()
        vsum, vn = 0.0, 0
        cell_correct = 0
        cell_total = 0
        stone_correct = 0
        stone_total = 0
        with torch.no_grad():
            for img, grid, mask in val_loader:
                img = img.to(device); grid = grid.to(device); mask = mask.to(device)
                logits = model(img)
                per_cell = ce(logits, grid)
                loss = (per_cell * mask).sum() / mask.sum().clamp(min=1.0)
                vsum += loss.item() * img.size(0); vn += img.size(0)
                preds = logits.argmax(dim=1)
                m = mask.bool()
                cell_correct += ((preds == grid) & m).sum().item()
                cell_total += m.sum().item()
                stone_mask = m & (grid > 0)  # only stone cells
                stone_correct += ((preds == grid) & stone_mask).sum().item()
                stone_total += stone_mask.sum().item()
        vloss = vsum / max(1, vn)
        cell_acc = cell_correct / max(1, cell_total)
        stone_acc = stone_correct / max(1, stone_total)

        marker = ""
        if vloss < best_val:
            best_val = vloss
            epochs_since_best = 0
            torch.save({"model": model.state_dict(), "val_loss": vloss}, args.model_path)
            marker = "  [best → saved]"
        else:
            epochs_since_best += 1
        print(
            f"epoch {epoch:>3d}  train {tloss:.4f}  val {vloss:.4f}  "
            f"cell_acc {cell_acc:.4f}  stone_acc {stone_acc:.4f}{marker}",
            flush=True,
        )
        if epochs_since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs", flush=True)
            break

    print(f"best val loss: {best_val:.4f} → {args.model_path}", flush=True)


if __name__ == "__main__":
    main()
