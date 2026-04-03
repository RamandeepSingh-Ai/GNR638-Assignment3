"""
Simple CNN backbone for Mask R-CNN.

Implements a minimal 3-layer convolutional backbone that extracts
hierarchical features from input images, producing a single feature
map with stride-8 and 256 channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Simplified 3-layer CNN backbone for feature extraction.

    Architecture:
        Input (B, 3, H, W)
        -> Conv(3, 64) + BN + ReLU + MaxPool(2)   # stride 2
        -> Conv(64, 128) + BN + ReLU + MaxPool(2)  # stride 4
        -> Conv(128, 256) + BN + ReLU + MaxPool(2) # stride 8
        Output (B, 256, H/8, W/8)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256) -> None:
        super(SimpleCNN, self).__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive. "
                f"Got in_channels={in_channels}, out_channels={out_channels}"
            )

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        assert x.size(1) == 3, f"Expected 3 input channels, got {x.size(1)}"

        x = self.pool1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool2(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = self.pool3(F.relu(self.bn3(self.conv3(x)), inplace=True))

        return x

    def get_out_channels(self) -> int:
        return 256

    def get_stride(self) -> int:
        return 8


def build_backbone(in_channels: int = 3, out_channels: int = 256) -> SimpleCNN:
    return SimpleCNN(in_channels=in_channels, out_channels=out_channels)
