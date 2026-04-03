"""
Region Proposal Network (RPN) implementation for Mask R-CNN.

Generates region proposals by predicting objectness scores and bounding
box adjustments for predefined anchor boxes at each feature map location.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class AnchorGenerator(nn.Module):
    """
    Generate anchor boxes for the RPN.

    Creates a grid of anchor boxes at multiple scales and aspect ratios.
    For each feature map location, K = len(scales) * len(aspect_ratios)
    anchors are generated (default K=9).
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (8, 16, 32),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        strides: Tuple[int, ...] = (8,),
    ) -> None:
        super(AnchorGenerator, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.num_anchors_per_location = len(scales) * len(aspect_ratios)

    def forward(
        self, feature_maps: List[torch.Tensor], image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate anchors for a given feature map and image size.

        Args:
            feature_maps: List of (B, C, H, W) feature tensors.
            image_size: (height, width) of input image.

        Returns:
            (num_anchors, 4) tensor in [x1, y1, x2, y2] format.
        """
        feature_tensor = feature_maps[0]
        fh = feature_tensor.size(2)
        fw = feature_tensor.size(3)
        stride = self.strides[0]
        device = feature_tensor.device
        dtype = feature_tensor.dtype

        anchors_list = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                for fy in range(fh):
                    for fx in range(fw):
                        cx = (fx + 0.5) * stride
                        cy = (fy + 0.5) * stride
                        anchors_list.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        if not anchors_list:
            return torch.zeros((0, 4), device=device, dtype=dtype)

        return torch.tensor(anchors_list, device=device, dtype=dtype)


class RPNHead(nn.Module):
    """
    RPN classification and bounding box regression head.

    Operates on feature maps to produce:
    - Objectness scores per anchor
    - Box deltas (tx, ty, tw, th) per anchor
    """

    def __init__(
        self, in_channels: int = 256, num_anchors: int = 9, hidden_channels: int = 256
    ) -> None:
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1, bias=True)
        self.cls_logits = nn.Conv2d(hidden_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(hidden_channels, num_anchors * 4, 1)
        self.num_anchors = num_anchors
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv(features), inplace=True)
        return self.cls_logits(x), self.bbox_pred(x)


class RPN(nn.Module):
    """
    Complete Region Proposal Network.

    Combines AnchorGenerator and RPNHead to produce filtered, NMS-deduplicated
    region proposals. Optionally computes RPN losses during training.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,
        anchor_scales: Tuple[int, ...] = (8, 16, 32),
        anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        strides: Tuple[int, ...] = (8,),
        pre_nms_top_k: int = 500,
        post_nms_top_k: int = 50,
        nms_thresh: float = 0.7,
    ) -> None:
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales, aspect_ratios=anchor_ratios, strides=strides
        )
        self.head = RPNHead(in_channels=in_channels, num_anchors=num_anchors)
        self.pre_nms_top_k = pre_nms_top_k
        self.post_nms_top_k = post_nms_top_k
        self.nms_thresh = nms_thresh
        self.num_anchors = num_anchors

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
        gt_boxes: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not features or not isinstance(features, list):
            raise ValueError("features must be a non-empty list")

        anchors = self.anchor_generator(features, image_size)
        cls_logits, bbox_pred = self.head(features[0])

        batch_size = cls_logits.size(0)
        cls_flat = cls_logits.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        bbox_flat = bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        scores = torch.sigmoid(cls_flat[0])
        proposals = self._generate_proposals(anchors, bbox_flat[:anchors.size(0)], scores)

        loss = None
        if training and gt_boxes is not None:
            loss = self._compute_loss(cls_flat[0], bbox_flat[:anchors.size(0)], anchors, gt_boxes)

        return proposals, loss

    def _generate_proposals(
        self,
        anchors: torch.Tensor,
        bbox_deltas: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        proposals = self._apply_box_deltas(anchors, bbox_deltas)
        keep_idx = torch.argsort(scores, descending=True)[:self.pre_nms_top_k]
        proposals = proposals[keep_idx]
        scores_k = scores[keep_idx]
        keep_nms = self._nms(proposals, scores_k, self.nms_thresh)
        proposals = proposals[keep_nms]
        return proposals[:self.post_nms_top_k]

    @staticmethod
    def _apply_box_deltas(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx, dy = deltas[:, 0], deltas[:, 1]
        dw = deltas[:, 2].clamp(max=4.0)
        dh = deltas[:, 3].clamp(max=4.0)

        pred_cx = ctr_x + dx * widths
        pred_cy = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)

        x1 = pred_cx - 0.5 * pred_w
        y1 = pred_cy - 0.5 * pred_h
        x2 = pred_cx + 0.5 * pred_w
        y2 = pred_cy + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float = 0.7) -> torch.Tensor:
        if boxes.size(0) == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        try:
            from torchvision.ops import nms
            return nms(boxes.float(), scores.float(), thresh)
        except ImportError:
            pass

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.argsort(scores, descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.max(x1[i], x1[rest])
            yy1 = torch.max(y1[i], y1[rest])
            xx2 = torch.min(x2[i], x2[rest])
            yy2 = torch.min(y2[i], y2[rest])
            inter = (xx2 - xx1 + 1).clamp(min=0) * (yy2 - yy1 + 1).clamp(min=0)
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)
            order = rest[iou <= thresh]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _compute_loss(
        self,
        cls_logits: torch.Tensor,
        bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if gt_boxes.numel() == 0:
            return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

        num_anchors = anchors.size(0)

        # Compute IoU between anchors and GT boxes for simple label assignment
        ious = self._compute_iou_matrix(anchors, gt_boxes)  # (num_anchors, num_gt)
        max_iou_per_anchor, _ = ious.max(dim=1)

        # Assign labels: positive >= 0.7, negative < 0.3, ignore in between
        labels = torch.full((num_anchors,), -1, dtype=torch.float32, device=cls_logits.device)
        labels[max_iou_per_anchor >= 0.7] = 1.0
        labels[max_iou_per_anchor < 0.3] = 0.0

        # Only compute loss on valid (non-ignored) anchors
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[valid_mask], labels[valid_mask], reduction='mean'
        )
        bbox_loss = F.smooth_l1_loss(bbox_deltas, torch.zeros_like(bbox_deltas), reduction='mean')

        return cls_loss + 0.5 * bbox_loss

    @staticmethod
    def _compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter

        return inter / union.clamp(min=1e-6)
