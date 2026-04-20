"""Side-by-side comparison of two stone-detector checkpoints on real crops.

For each real crop under training_data/stone_points/ (the user's manually
labeled book crops), run both models and emit a single PNG with the crop,
each model's overlaid detections, and confidence labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .stone_inference import IMG_SIZE, _extract_peaks, _estimate_display_radius
from .train_stones import UNet


ROOT = Path(__file__).resolve().parents[2]


def load(model_path: Path) -> tuple[UNet, torch.device]:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = UNet()
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, device


def detect(model: UNet, device: torch.device, crop_bgr: np.ndarray,
           peak_thresh: float = 0.3) -> list[dict]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = (torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        heat = model(tensor)[0].cpu().numpy()
    h, w = crop_bgr.shape[:2]
    sx, sy = w / IMG_SIZE, h / IMG_SIZE
    nms_r = max(3, int(IMG_SIZE * 0.02))
    raw: list[dict] = []
    for ch, color in enumerate(("B", "W")):
        for px, py, conf in _extract_peaks(heat[ch], peak_thresh, nms_r):
            raw.append({"x": px * sx, "y": py * sy, "color": color, "conf": conf})
    r = _estimate_display_radius(raw, fallback=min(h, w) * 0.03, image_max=min(h, w))
    for d in raw:
        d["r"] = r
    # Mirror production cross-channel dedup (see stone_inference.detect_stones_cnn).
    raw.sort(key=lambda s: -s["conf"])
    deduped: list[dict] = []
    merge_dist = max(r * 1.6, min(h, w) * 0.035, nms_r * max(sx, sy))
    for d in raw:
        if any(
            (d["x"] - m["x"]) ** 2 + (d["y"] - m["y"]) ** 2 < merge_dist ** 2
            for m in deduped
        ):
            continue
        deduped.append(d)
    return deduped


def overlay(img_bgr: np.ndarray, dets: list[dict]) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for d in dets:
        r = d["r"]
        col = (0, 200, 0) if d["color"] == "B" else (255, 100, 0)
        draw.ellipse(
            [(d["x"] - r, d["y"] - r), (d["x"] + r, d["y"] + r)],
            outline=col, width=2,
        )
    return pil


def side_by_side(
    crop: np.ndarray, real_dets: list[dict], synth_dets: list[dict], title: str,
) -> Image.Image:
    a = overlay(crop, real_dets)
    b = overlay(crop, synth_dets)
    pad = 20
    label_h = 28
    w = a.width + b.width + pad * 3
    h = max(a.height, b.height) + pad * 2 + label_h
    out = Image.new("RGB", (w, h), (240, 240, 240))
    out.paste(a, (pad, pad + label_h))
    out.paste(b, (a.width + pad * 2, pad + label_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((pad, pad // 2), f"REAL-trained  ({title})", fill=(20, 20, 20), font=font)
    draw.text((a.width + pad * 2, pad // 2), "SYNTH-trained",
              fill=(20, 20, 20), font=font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-model", type=Path,
                    default=ROOT / "models" / "stone_detector.pt")
    ap.add_argument("--synth-model", type=Path,
                    default=ROOT / "models" / "stone_detector_synth.pt")
    ap.add_argument("--crops-dir", type=Path,
                    default=ROOT / "training_data" / "stone_points")
    ap.add_argument("--out", type=Path, default=Path("/tmp/synth_eval"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    real_m, dev = load(args.real_model)
    synth_m, _ = load(args.synth_model)

    pngs = sorted(args.crops_dir.glob("*.png"))[:args.n]
    if not pngs:
        raise SystemExit(f"no crops in {args.crops_dir}")

    print(f"comparing on {len(pngs)} real crops")
    for png in pngs:
        arr = np.frombuffer(png.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        real_d = detect(real_m, dev, img)
        synth_d = detect(synth_m, dev, img)
        panel = side_by_side(img, real_d, synth_d, png.stem)
        out_path = args.out / f"{png.stem}.png"
        panel.save(out_path)
        print(f"  {out_path}  real={len(real_d)}  synth={len(synth_d)}")


if __name__ == "__main__":
    main()
