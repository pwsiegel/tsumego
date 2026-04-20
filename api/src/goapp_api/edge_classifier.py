"""Small CNN that predicts the 4 edge-is-boundary bits for a board crop.

Input:  grayscale crop resized to 256x256
Output: 4 logits (left, right, top, bottom) → sigmoid → probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn


IMG_SIZE = 256


class EdgeClassifier(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16):
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

        self.features = nn.Sequential(
            block(in_ch, base),        # 256 -> 128
            block(base, base * 2),     # 128 -> 64
            block(base * 2, base * 4), # 64 -> 32
            block(base * 4, base * 8), # 32 -> 16
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


EDGE_NAMES = ("left", "right", "top", "bottom")
