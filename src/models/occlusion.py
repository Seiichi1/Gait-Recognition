"""
Occlusion-aware masking modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OcclusionMaskEstimator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(16, in_channels // 4)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_head(x)


class PartVisibilityScorer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        hidden = max(32, feature_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        """Score part visibility.

        Args:
            part_features: Tensor with shape ``(B, T, P, D)``.
        """
        b, t, p, d = part_features.shape
        scores = self.net(part_features.reshape(b * t * p, d))
        return scores.reshape(b, t, p)
