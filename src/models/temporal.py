"""
Temporal aggregation modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is None:
            return x.mean(dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TemporalGRUAggregator(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gru = nn.GRU(feature_dim, feature_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.gru(x)
        if weights is None:
            pooled = y.mean(dim=1)
        else:
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            pooled = torch.sum(y * weights.unsqueeze(-1), dim=1)
        return self.proj(pooled)


def build_temporal_module(module_type: str, feature_dim: int) -> nn.Module:
    normalized = (module_type or "mean").lower()
    if normalized in {"convlstm", "bi_convlstm", "tcn", "transformer"}:
        return TemporalGRUAggregator(feature_dim)
    return TemporalMeanPooling()
