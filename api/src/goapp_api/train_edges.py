"""Train the 4-edge-bit classifier on synth edge crops.

Usage:
    uv --directory api run --extra ml python -m goapp_api.train_edges \\
        --data-dir <path> --model-path <path> --epochs 10 --batch-size 32
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

from .edge_classifier import EDGE_NAMES, IMG_SIZE, EdgeClassifier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "training_data" / "synth_edge_crops"
DEFAULT_MODEL = ROOT / "models" / "edge_classifier.pt"
SEED = 42


@dataclass(frozen=True)
class Example:
    image_path: Path
    label_path: Path


class EdgeDataset(Dataset):
    def __init__(self, examples: list[Example], augment: bool = False):
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = cv2.imread(str(ex.image_path), cv2.IMREAD_GRAYSCALE)
        label = json.loads(ex.label_path.read_text())
        bits = {k: bool(label.get(k, False)) for k in EDGE_NAMES}

        if self.augment:
            # Bbox-slack jitter: extend or trim up to ±8% on each side,
            # simulating the looser bboxes a production YOLO may produce.
            h, w = img.shape
            jt = int(np.random.uniform(-0.08, 0.08) * h)
            jb = int(np.random.uniform(-0.08, 0.08) * h)
            jl = int(np.random.uniform(-0.08, 0.08) * w)
            jr = int(np.random.uniform(-0.08, 0.08) * w)
            img = cv2.copyMakeBorder(
                img,
                max(0, -jt), max(0, -jb), max(0, -jl), max(0, -jr),
                cv2.BORDER_CONSTANT, value=245,
            )
            y0 = max(0, jt); x0 = max(0, jl)
            y1 = img.shape[0] - max(0, jb)
            x1 = img.shape[1] - max(0, jr)
            if y1 > y0 and x1 > x0:
                img = img[y0:y1, x0:x1]
            # Horizontal flip (swaps left/right bits).
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                bits["left"], bits["right"] = bits["right"], bits["left"]
            # Vertical flip (swaps top/bottom bits).
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :])
                bits["top"], bits["bottom"] = bits["bottom"], bits["top"]
            # Light brightness jitter.
            alpha = 0.9 + random.random() * 0.2
            beta = (random.random() - 0.5) * 20
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        target = torch.tensor(
            [float(bits[k]) for k in EDGE_NAMES], dtype=torch.float32,
        )
        return tensor, target


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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    examples = gather(args.data_dir)
    if not examples:
        raise RuntimeError(f"no edge-labeled crops in {args.data_dir}")

    random.seed(SEED)
    random.shuffle(examples)
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]
    split = max(1, int(len(examples) * (1 - args.val_frac)))
    train_ex, val_ex = examples[:split], examples[split:]
    print(f"examples: {len(examples)} total → {len(train_ex)} train / {len(val_ex)} val")
    print(f"data: {args.data_dir}")
    print(f"model out: {args.model_path}")

    device = pick_device()
    print(f"device: {device}")

    train_loader = DataLoader(
        EdgeDataset(train_ex, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        EdgeDataset(val_ex, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = EdgeClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tsum, tn = 0.0, 0
        for img, target in train_loader:
            img = img.to(device); target = target.to(device)
            logit = model(img)
            loss = loss_fn(logit, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tsum += loss.item() * img.size(0); tn += img.size(0)
        tloss = tsum / max(1, tn)

        model.eval()
        vsum, vn, correct = 0.0, 0, 0
        with torch.no_grad():
            for img, target in val_loader:
                img = img.to(device); target = target.to(device)
                logit = model(img)
                vsum += loss_fn(logit, target).item() * img.size(0)
                vn += img.size(0)
                preds = (logit.sigmoid() > 0.5).float()
                correct += (preds == target).all(dim=1).sum().item()
        vloss = vsum / max(1, vn)
        acc = correct / max(1, vn)

        marker = ""
        if vloss < best_val:
            best_val = vloss
            epochs_since_best = 0
            torch.save({"model": model.state_dict(), "val_loss": vloss}, args.model_path)
            marker = "  [best → saved]"
        else:
            epochs_since_best += 1
        print(f"epoch {epoch:>3d}  train {tloss:.4f}  val {vloss:.4f}  "
              f"all-4-correct {acc:.3f}{marker}")
        if epochs_since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs")
            break

    print(f"best val loss: {best_val:.4f} → {args.model_path}")


if __name__ == "__main__":
    main()
