"""
ROI Align implementation using bilinear interpolation.

ROI Align extracts aligned features from Region of Interests (RoIs)
using bilinear interpolation to maintain spatial precision, which is
critical for Mask R-CNN mask quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ROIAlign(nn.Module):
    """
    ROI Align layer for extracting features from regions of interest.

    Key advantage over ROI Pooling: no coordinate quantization.
    Bilinear interpolation preserves gradients and improves mask quality.

    Args:
        output_size: H and W of output feature maps (e.g. 7 for 7x7).
        spatial_scale: Scale from image coords to feature map coords.
            For stride-8 features: spatial_scale = 1/8 = 0.125.
        sampling_ratio: Sampling points per bin (0 = adaptive).
    """

    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 0.125,
        sampling_ratio: int = 0,
    ) -> None:
        super(ROIAlign, self).__init__()

        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if spatial_scale <= 0:
            raise ValueError(f"spatial_scale must be positive, got {spatial_scale}")

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Extract aligned features from ROIs using grid_sample for efficiency.

        Args:
            features: (B, C, H, W) feature maps.
            rois: (N, 4) bounding boxes in image space [x1, y1, x2, y2].

        Returns:
            (N, C, output_size, output_size) aligned feature maps.
        """
        assert features.dim() == 4
        assert rois.dim() == 2 and rois.size(1) == 4

        num_rois = rois.size(0)
        feature_h = features.size(2)
        feature_w = features.size(3)

        # Scale ROIs to feature map space
        rois_feat = rois * self.spatial_scale

        # Build normalized [-1, 1] sampling grids for each ROI
        # Use batch dimension 0 (single image assumed per call)
        grids = []
        for i in range(num_rois):
            x1, y1, x2, y2 = rois_feat[i].detach()
            x1 = x1.clamp(0, feature_w - 1)
            y1 = y1.clamp(0, feature_h - 1)
            x2 = x2.clamp(0, feature_w - 1)
            y2 = y2.clamp(0, feature_h - 1)

            # Create grid of shape (output_size, output_size, 2)
            ys = torch.linspace(float(y1), float(y2), self.output_size, device=features.device)
            xs = torch.linspace(float(x1), float(x2), self.output_size, device=features.device)

            # Normalize to [-1, 1]
            xs_norm = (xs / (feature_w - 1)) * 2 - 1
            ys_norm = (ys / (feature_h - 1)) * 2 - 1

            grid_x, grid_y = torch.meshgrid(xs_norm, ys_norm, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_out, W_out, 2)
            grids.append(grid)

        grids = torch.stack(grids, dim=0)  # (N, H_out, W_out, 2)

        # Expand features to match N ROIs (all from batch index 0)
        features_expanded = features[0:1].expand(num_rois, -1, -1, -1)  # (N, C, H, W)

        output = F.grid_sample(
            features_expanded,
            grids,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # (N, C, output_size, output_size)

        return output


def roi_align(
    features: torch.Tensor,
    rois: torch.Tensor,
    output_size: int = 7,
    spatial_scale: float = 0.125,
    sampling_ratio: int = 0,
) -> torch.Tensor:
    """Functional interface for ROI Align."""
    op = ROIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    return op(features, rois)
