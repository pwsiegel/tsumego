"""Train the grid-geometry regressor on per-board crops from synth pages.

Usage:
    uv --directory backend run --extra ml python -m goapp.ml.grid_detect.train \\
        --device mps --limit 1500 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import IMG_SIZE, GridCropsDataset
from .model import GridDetector

from ...paths import (
    GRID_DETECTOR_PATH,
    MODELS_DIR,
    SYNTH_PAGES_DIR as DEFAULT_PAGES,
    TRAINING_RUNS_DIR,
)


SEED = 42
VAL_FRAC = 0.1
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
NUM_WORKERS = 4

RUN_NAME = "grid_detector"


def _split_pages(pages_dir: Path, limit: int | None) -> tuple[list[Path], list[Path]]:
    page_jsons = sorted(pages_dir.glob("*.json"))
    if limit:
        page_jsons = page_jsons[:limit]
    if not page_jsons:
        raise RuntimeError(f"no page JSONs in {pages_dir}")
    rng = random.Random(SEED)
    shuffled = page_jsons.copy()
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - VAL_FRAC)))
    return shuffled[:split_at], shuffled[split_at:]


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(imgs)
                loss = loss_fn(preds, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
    return total_loss / max(1, n_seen)


def train(
    pages_dir: Path,
    limit: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    model_out: Path,
    device_arg: str | None,
) -> Path:
    train_pages, val_pages = _split_pages(pages_dir, limit)
    train_ds = GridCropsDataset(train_pages, augment=True)
    val_ds = GridCropsDataset(val_pages, augment=False)
    print(f"dataset: {len(train_ds)} train / {len(val_ds)} val crops "
          f"from {len(train_pages)} train / {len(val_pages)} val pages")

    device = _select_device(device_arg)
    print(f"device: {device}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=NUM_WORKERS > 0,
    )

    model = GridDetector(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    run_dir = TRAINING_RUNS_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "results.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    best_val = float("inf")
    best_path = run_dir / "best.pt"
    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_loss = _epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = _epoch(model, val_loader, loss_fn, device, None)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  epoch {ep:3d}/{epochs}  "
              f"train {train_loss:.5f}  val {val_loss:.5f}  "
              f"lr {cur_lr:.2e}  ({elapsed:.0f}s)")
        with log_path.open("a") as f:
            f.write(f"{ep},{train_loss:.6f},{val_loss:.6f},{cur_lr:.6e}\n")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(best_path, map_location="cpu"), model_out)
    metadata = {
        "img_size": IMG_SIZE,
        "best_val_loss": best_val,
        "epochs": epochs,
    }
    (run_dir / "meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"best val loss {best_val:.5f} → {model_out}")
    return model_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    ap.add_argument("--limit", type=int, default=1500,
                    help="number of synth pages to process")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--model-out", type=Path, default=GRID_DETECTOR_PATH)
    ap.add_argument("--device", type=str, default=None,
                    help='"cpu", "mps", "cuda", "0", etc. Default auto-selects.')
    args = ap.parse_args()
    train(
        args.pages, args.limit, args.epochs,
        args.batch_size, args.lr,
        args.model_out, args.device,
    )


if __name__ == "__main__":
    main()
