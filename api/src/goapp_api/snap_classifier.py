"""Small CNN that predicts the grid transformation (pitch, origin) for a
tight-ish board crop.

Given a YOLO-produced crop resized to IMG_SIZE, outputs four scalars —
(pitch_x_frac, pitch_y_frac, origin_x_frac, origin_y_frac) — all expressed
as FRACTIONS of the crop width/height. Applied to a stone's pixel position
(x_frac, y_frac):

    col_local = round((x_frac - origin_x_frac) / pitch_x_frac)

...then add col_min from the edge classifier to get the absolute column.

Fractions keep the output scale-invariant so the same model handles full
boards, corners, sides, and mid-board crops with different effective pitches.
"""

from __future__ import annotations

import torch
import torch.nn as nn


IMG_SIZE = 256


class SnapToGrid(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.features = nn.Sequential(
            block(in_ch, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8),
            block(base * 8, base * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 4): (pitch_x_frac, pitch_y_frac, origin_x_frac,
        origin_y_frac)."""
        return self.head(self.features(x))
