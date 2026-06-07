"""
Baseline gait recognition model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .attention import build_attention
from .backbone import build_backbone


class HorizontalPyramidPooling(nn.Module):
    def __init__(self, in_channels: int, num_parts: int, feature_dim: int):
        super().__init__()
        self.num_parts = num_parts
        self.part_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channels, feature_dim), nn.ReLU(inplace=True)) for _ in range(num_parts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        part_height = max(1, h // self.num_parts)
        part_features: List[torch.Tensor] = []

        for idx in range(self.num_parts):
            start = idx * part_height
            end = h if idx == self.num_parts - 1 else min(h, (idx + 1) * part_height)
            part = x[:, :, :, start:end, :]
            pooled = part.mean(dim=(-1, -2))
            part_features.append(self.part_mlps[idx](pooled))

        return torch.stack(part_features, dim=2)


class BaselineGaitModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        attention_cfg = model_cfg.get("attention", {})
        self.backbone, out_channels = build_backbone(
            name=model_cfg.get("backbone", "resnet50"),
            input_channels=model_cfg.get("input_channels", 1),
            pretrained=model_cfg.get("pretrained", False),
        )
        self.attention = build_attention(
            attention_cfg.get("type", "none"),
            out_channels,
            attention_cfg.get("reduction_ratio", 16),
        )
        self.hpp = HorizontalPyramidPooling(
            in_channels=out_channels,
            num_parts=model_cfg.get("num_parts", 4),
            feature_dim=model_cfg.get("feature_dim", 256),
        )

    def forward(self, silhouettes: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = silhouettes.shape
        features = self.backbone(silhouettes.reshape(b * t, c, h, w))
        features = self.attention(features)
        _, feat_c, feat_h, feat_w = features.shape
        features = features.reshape(b, t, feat_c, feat_h, feat_w)
        part_features = self.hpp(features)
        return part_features.mean(dim=1)
