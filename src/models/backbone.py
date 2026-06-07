"""
Backbone networks for gait recognition.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleBackbone(nn.Module):
    """Compact CNN backbone that works without torchvision."""

    def __init__(self, input_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(base_channels, base_channels)
        self.layer2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.out_channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def adapt_first_conv(conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    if conv.in_channels == input_channels:
        return conv

    adapted = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        if conv.weight.shape[1] >= input_channels:
            adapted.weight.copy_(conv.weight[:, :input_channels])
        else:
            adapted.weight.copy_(conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1))
        if conv.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


def build_backbone(
    name: str = "resnet50",
    input_channels: int = 1,
    pretrained: bool = False,
) -> Tuple[nn.Module, int]:
    """Build a feature extractor.

    Falls back to a compact native CNN if torchvision is unavailable.
    """
    normalized = (name or "resnet50").lower()

    if normalized in {"resnet50", "efficientnet_b0", "efficientnet-b0"}:
        try:
            import torchvision.models as tv_models

            if normalized == "resnet50":
                model = tv_models.resnet50(weights=None if not pretrained else tv_models.ResNet50_Weights.DEFAULT)
                model.conv1 = adapt_first_conv(model.conv1, input_channels)
                layers = [
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                ]
                backbone = nn.Sequential(*layers)
                out_channels = 2048
                return backbone, out_channels

            model = tv_models.efficientnet_b0(
                weights=None if not pretrained else tv_models.EfficientNet_B0_Weights.DEFAULT
            )
            first_conv = model.features[0][0]
            model.features[0][0] = adapt_first_conv(first_conv, input_channels)
            return model.features, 1280
        except Exception:
            pass

    fallback = SimpleBackbone(input_channels=input_channels, base_channels=32)
    return fallback, fallback.out_channels
