"""
Attention blocks for gait recognition.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def build_attention(attention_type: str, channels: int, reduction_ratio: int = 16) -> nn.Module:
    normalized = (attention_type or "none").lower()
    if normalized == "se":
        return SEBlock(channels, reduction_ratio)
    if normalized == "cbam":
        return CBAM(channels, reduction_ratio)
    return nn.Identity()
