"""CNN that predicts a 19x19x3 (empty/black/white) grid from a board crop.

Architecture: a small conv encoder that reduces 304x304 to 19x19, plus a
global-context pathway (global average pool → project → broadcast) that
concatenates onto every output cell. Each output cell therefore sees both
its own 16x16 local patch *and* a summary of the whole input — which is
needed because the relationship between input pixels and board cells varies
with the crop's visible-window size.

Input:  grayscale crop resized to 304x304
Output: logits tensor of shape (batch, 3, 19, 19)
"""

from __future__ import annotations

import torch
import torch.nn as nn


IMG_SIZE = 304  # 19 * 16 — after 4 max-pool-2 blocks the feature map is 19x19.
BOARD_SIZE = 19
N_CLASSES = 3  # 0=empty, 1=B, 2=W


class GridClassifier(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 48):
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

        # 304 -> 152 -> 76 -> 38 -> 19
        self.features = nn.Sequential(
            block(in_ch, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8),
        )
        enc_ch = base * 8

        # Global context: average-pool the whole encoder output to one
        # vector, project it, then broadcast back to a 19x19 feature map.
        self.global_proj = nn.Sequential(
            nn.Conv2d(enc_ch, enc_ch, 1),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
        )

        # Per-cell head sees concat(local features, global features).
        self.head = nn.Sequential(
            nn.Conv2d(enc_ch * 2, enc_ch, 1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, N_CLASSES, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                  # (B, enc_ch, 19, 19)
        ctx = feat.mean(dim=(2, 3), keepdim=True)  # (B, enc_ch, 1, 1)
        ctx = self.global_proj(ctx)              # (B, enc_ch, 1, 1)
        ctx = ctx.expand(-1, -1, BOARD_SIZE, BOARD_SIZE)  # broadcast to 19x19
        combined = torch.cat([feat, ctx], dim=1)  # (B, 2*enc_ch, 19, 19)
        logits = self.head(combined)             # (B, 3, 19, 19)
        return logits

    @staticmethod
    def predict_grid(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)
