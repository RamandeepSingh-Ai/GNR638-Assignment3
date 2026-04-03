"""
Detection heads for Mask R-CNN — matching paper Figure 4 (FPN variant).

Paper architecture (right panel of Figure 4):
  Box/Cls head:  7x7 RoI features → flatten → fc(1024) → fc(1024) → cls/box
  Mask head:     14x14 RoI features → 4 × [3x3 conv 256] → deconv 2x2 → 1x1 conv → K masks 28x28

We use 7x7 pool for cls/box, and 14x14 pool for mask (as in the paper).
For simplicity we use a single 7x7 pool and let the mask head upsample via deconv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DetectionHeads(nn.Module):
    """Paper-faithful FPN-style heads (Figure 4 right)."""

    def __init__(
        self,
        in_channels: int = 256,
        pool_size: int = 7,
        num_classes: int = 5,
        fc_channels: int = 1024,
        mask_output_size: int = 28,
    ):
        super().__init__()
        self.num_classes = num_classes
        flat = in_channels * pool_size * pool_size

        # Box/Cls head: shared FC trunk (paper: two fc1024 layers)
        self.fc1 = nn.Linear(flat, fc_channels)
        self.fc2 = nn.Linear(fc_channels, fc_channels)
        self.cls_score = nn.Linear(fc_channels, num_classes)
        self.bbox_pred = nn.Linear(fc_channels, num_classes * 4)

        # Mask head (paper Figure 4 right): 4 × conv3x3(256) → deconv → deconv → conv1x1
        # From 7x7 pool: 7→14 (deconv1) →28 (deconv2) → 28x28 × K masks
        self.mask_convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.mask_deconv1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_deconv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for dc in [self.mask_deconv1, self.mask_deconv2]:
            nn.init.kaiming_normal_(dc.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(dc.bias, 0)

    def forward(self, roi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = roi_features.view(roi_features.size(0), -1)
        fc = F.relu(self.fc1(x_flat), inplace=True)
        fc = F.relu(self.fc2(fc), inplace=True)
        cls_logits = self.cls_score(fc)
        bbox_deltas = self.bbox_pred(fc)

        mask_feat = self.mask_convs(roi_features)
        mask_feat = F.relu(self.mask_deconv1(mask_feat), inplace=True)
        mask_feat = F.relu(self.mask_deconv2(mask_feat), inplace=True)
        mask_logits = self.mask_pred(mask_feat)

        return cls_logits, bbox_deltas, mask_logits
