"""
Feature fusion helpers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OcclusionConditionedFusion(nn.Module):
    def __init__(self, feature_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, part_features: torch.Tensor, visibility: torch.Tensor | None = None) -> torch.Tensor:
        if visibility is None:
            fused = part_features.mean(dim=1)
        else:
            weights = visibility / (visibility.sum(dim=1, keepdim=True) + 1e-8)
            fused = torch.sum(part_features * weights.unsqueeze(-1), dim=1)
        return self.proj(fused)
