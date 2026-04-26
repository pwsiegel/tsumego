"""ResNet18 regressor that predicts grid geometry from a board crop.

Output is 6 numbers in [0, 1]:
    gx1, gy1, gx2, gy2  — bbox of the grid INTERSECTION rectangle
                          (top-left intersection to bottom-right
                          intersection), as a fraction of input width/height.
                          Note: this is a tighter rectangle than the visual
                          frame, which sits ~half a pitch outside.
    px, py              — pitch (cell spacing) as a fraction of input
                          width and height respectively.

Edges are derived post-hoc from the bbox: a side is a real boundary iff
its margin to the corresponding crop edge is < ~0.5 * pitch.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


N_OUTPUTS = 6


class GridDetector(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, N_OUTPUTS),
            nn.Sigmoid(),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)
