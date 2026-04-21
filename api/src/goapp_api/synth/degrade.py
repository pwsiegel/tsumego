"""Apply scan-like degradations to a synthetic page while keeping the
board bboxes and stone pixel coords consistent.

Operations:
  - Small rotation (±2°)
  - Gaussian blur (σ 0–1.2)
  - Additive gaussian noise
  - Contrast + brightness jitter
  - Optional JPEG recompression artifact

The rotation is the only one that changes geometry; we recompute every
annotated point/bbox through the same rotation matrix so labels match.
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageFilter

from .page_compose import BoardAnnotation, Page


def degrade(page: Page, rng: random.Random) -> Page:
    img = page.image
    boards = page.boards

    # ---- rotation ----
    angle = rng.uniform(-2.0, 2.0)
    if abs(angle) > 0.05:
        img, boards = _rotate(img, boards, angle, bg=(252, 248, 236))

    # ---- blur ----
    sigma = rng.uniform(0.0, 1.2)
    if sigma > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # ---- noise ----
    arr = np.array(img, dtype=np.float32)
    noise_std = rng.uniform(0.0, 6.0)
    if noise_std > 0.1:
        arr += rng.gauss(0, 1) * 0  # silence warning; branch below adds noise
        arr += np.random.normal(0, noise_std, arr.shape).astype(np.float32)

    # ---- contrast / brightness ----
    contrast = rng.uniform(0.9, 1.1)
    brightness = rng.uniform(-8, 8)
    arr = np.clip((arr - 128) * contrast + 128 + brightness, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # ---- optional JPEG recompression ----
    if rng.random() < 0.5:
        quality = rng.randint(55, 90)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return Page(image=img, lang_code=page.lang_code, boards=boards)


def _rotate(
    img: Image.Image,
    boards: list[BoardAnnotation],
    angle_deg: float,
    bg: tuple[int, int, int],
) -> tuple[Image.Image, list[BoardAnnotation]]:
    """Rotate the page and propagate the rotation to every annotation."""
    W, H = img.size
    cx, cy = W / 2.0, H / 2.0
    rotated = img.rotate(
        angle_deg, resample=Image.BICUBIC, fillcolor=bg, expand=False,
    )

    # PIL rotates CCW for positive angles; to map an input point to its
    # output position we apply the same CCW rotation around the center.
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotate_point(x: float, y: float) -> tuple[float, float]:
        dx = x - cx
        dy = y - cy
        rx = cos_t * dx + sin_t * dy
        ry = -sin_t * dx + cos_t * dy
        return (cx + rx, cy + ry)

    def rotate_bbox(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        rot = [rotate_point(x, y) for x, y in corners]
        return (
            int(round(min(p[0] for p in rot))),
            int(round(min(p[1] for p in rot))),
            int(round(max(p[0] for p in rot))),
            int(round(max(p[1] for p in rot))),
        )

    new_boards: list[BoardAnnotation] = []
    for b in boards:
        tight = rotate_bbox(*b.bbox)
        padded = rotate_bbox(*b.bbox_padded)
        loose = rotate_bbox(*b.loose_bbox)
        new_stones = [
            (
                int(round(rotate_point(sx, sy)[0])),
                int(round(rotate_point(sx, sy)[1])),
                color,
            )
            for (sx, sy, color) in b.stone_centers
        ]
        new_boards.append(BoardAnnotation(
            bbox=tight,
            bbox_padded=padded,
            loose_bbox=loose,
            window=b.window,
            edges_on_board=b.edges_on_board,
            edge_class=b.edge_class,
            stone_centers=new_stones,
        ))
    return rotated, new_boards
