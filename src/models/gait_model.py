"""
Integrated occlusion-aware gait model.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .attention import build_attention
from .backbone import build_backbone
from .fusion import OcclusionConditionedFusion
from .occlusion import OcclusionMaskEstimator, PartVisibilityScorer
from .temporal import build_temporal_module


class FullGaitModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        attention_cfg = model_cfg.get("attention", {})
        self.num_parts = model_cfg.get("num_parts", 4)
        self.feature_dim = model_cfg.get("feature_dim", 256)

        self.backbone, out_channels = build_backbone(
            name=model_cfg.get("backbone", "resnet50"),
            input_channels=model_cfg.get("input_channels", 1),
            pretrained=model_cfg.get("pretrained", False),
        )
        self.attention = build_attention(
            attention_cfg.get("type", "cbam"),
            out_channels,
            attention_cfg.get("reduction_ratio", 16),
        )
        self.mask_estimator = OcclusionMaskEstimator(out_channels)
        self.part_projection = nn.ModuleList(
            [nn.Sequential(nn.Linear(out_channels, self.feature_dim), nn.ReLU(inplace=True)) for _ in range(self.num_parts)]
        )
        self.visibility_scorer = PartVisibilityScorer(self.feature_dim)
        self.temporal_module = build_temporal_module(
            model_cfg.get("temporal", {}).get("type", "bi_convlstm"),
            self.feature_dim,
        )
        self.fusion = OcclusionConditionedFusion(self.feature_dim, self.feature_dim)

    def _pool_parts(self, features: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = features.shape
        part_height = max(1, h // self.num_parts)
        pooled_parts = []
        for idx in range(self.num_parts):
            start = idx * part_height
            end = h if idx == self.num_parts - 1 else min(h, (idx + 1) * part_height)
            part = features[:, :, :, start:end, :]
            pooled = part.mean(dim=(-1, -2))
            pooled_parts.append(self.part_projection[idx](pooled))
        return torch.stack(pooled_parts, dim=2)

    def forward(self, silhouettes: torch.Tensor, return_aux: bool = False):
        b, t, c, h, w = silhouettes.shape
        features = self.backbone(silhouettes.reshape(b * t, c, h, w))
        features = self.attention(features)
        masks = self.mask_estimator(features)
        features = features * masks
        _, feat_c, feat_h, feat_w = features.shape
        features = features.reshape(b, t, feat_c, feat_h, feat_w)
        masks = masks.reshape(b, t, 1, feat_h, feat_w)

        part_features = self._pool_parts(features)
        part_visibility = self.visibility_scorer(part_features)

        aggregated_parts = []
        for idx in range(self.num_parts):
            aggregated_parts.append(self.temporal_module(part_features[:, :, idx, :], part_visibility[:, :, idx]))
        aggregated_parts = torch.stack(aggregated_parts, dim=1)

        embedding = self.fusion(aggregated_parts, aggregated_parts.norm(dim=-1))

        if not return_aux:
            return embedding

        return {
            "embedding": embedding,
            "part_features": aggregated_parts,
            "visibility": part_visibility,
            "masks": masks,
        }
