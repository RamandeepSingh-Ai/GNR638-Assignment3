"""
Mask R-CNN — faithful replication of He et al. ICCV 2017.

Training (Section 3.1):
  - GT boxes injected into proposals → IoU matching (>=0.5 positive, else negative)
  - Sample N=128 RoIs per image (1:3 positive:negative ratio)
  - L = L_cls + L_box + L_mask
  - L_mask: per-pixel sigmoid + binary cross-entropy on the k-th mask only (Eq in Sec 3)
  - Mask target: intersection of RoI with GT mask, resized to m×m

Inference (Section 3.1):
  - RPN → 300 proposals
  - Box branch on all proposals → per-class NMS (thresh 0.5)
  - Top 100 detections → mask branch
  - k-th mask (predicted class) → resize to RoI size → binarize at 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from backbone import build_backbone
from roi_align import ROIAlign
from rpn import RPN
from heads import DetectionHeads


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = a1[:, None] + a2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes, scores, thresh):
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), thresh)
    except ImportError:
        pass
    order = scores.argsort(descending=True)
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
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
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)
        order = rest[iou <= thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def encode_boxes(proposals, gt_boxes):
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1)
    pcx = proposals[:, 0] + 0.5 * pw
    pcy = proposals[:, 1] + 0.5 * ph
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1)
    gcx = gt_boxes[:, 0] + 0.5 * gw
    gcy = gt_boxes[:, 1] + 0.5 * gh
    return torch.stack([(gcx - pcx) / pw, (gcy - pcy) / ph,
                        torch.log(gw / pw), torch.log(gh / ph)], dim=1)


def decode_boxes(proposals, deltas):
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1)
    pcx = proposals[:, 0] + 0.5 * pw
    pcy = proposals[:, 1] + 0.5 * ph
    dx, dy = deltas[:, 0], deltas[:, 1]
    dw = deltas[:, 2].clamp(max=4.0)
    dh = deltas[:, 3].clamp(max=4.0)
    ncx = pcx + dx * pw
    ncy = pcy + dy * ph
    nw = pw * torch.exp(dw)
    nh = ph * torch.exp(dh)
    return torch.stack([ncx - 0.5 * nw, ncy - 0.5 * nh,
                        ncx + 0.5 * nw, ncy + 0.5 * nh], dim=1)


class SimpleMaskRCNN(nn.Module):
    """Mask R-CNN (He et al. 2017) — SimpleCNN backbone."""

    def __init__(
        self,
        num_classes: int = 5,
        backbone_out_channels: int = 256,
        backbone_stride: int = 8,
        roi_pool_size: int = 7,
        mask_output_size: int = 28,
        rpn_post_nms_top_k: int = 300,
        train_rois_per_image: int = 128,
        train_fg_fraction: float = 0.25,
        fg_iou_thresh: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_stride = backbone_stride
        self.roi_pool_size = roi_pool_size
        self.mask_output_size = mask_output_size
        self.rpn_post_nms_top_k = rpn_post_nms_top_k
        self.train_rois_per_image = train_rois_per_image
        self.train_fg_fraction = train_fg_fraction
        self.fg_iou_thresh = fg_iou_thresh

        self.backbone = build_backbone(out_channels=backbone_out_channels)
        self.rpn = RPN(
            in_channels=backbone_out_channels,
            num_anchors=15,
            anchor_scales=(4, 8, 16, 32, 64),
            anchor_ratios=(0.5, 1.0, 2.0),
            strides=(backbone_stride,),
            pre_nms_top_k=2000,
            post_nms_top_k=rpn_post_nms_top_k,
            nms_thresh=0.7,
        )
        self.roi_align = ROIAlign(output_size=roi_pool_size,
                                   spatial_scale=1.0 / backbone_stride)
        self.heads = DetectionHeads(
            in_channels=backbone_out_channels,
            pool_size=roi_pool_size,
            num_classes=num_classes,
            mask_output_size=mask_output_size,
        )

    def _sample_proposals(self, proposals, gt_boxes, gt_classes, gt_masks, device):
        ms = self.mask_output_size
        all_boxes = torch.cat([proposals, gt_boxes], dim=0)
        iou = box_iou(all_boxes, gt_boxes)
        max_iou, matched_gt = iou.max(dim=1)

        labels = gt_classes[matched_gt].long().to(device)
        labels[max_iou < self.fg_iou_thresh] = 0

        fg_idx = torch.where(labels > 0)[0]
        bg_idx = torch.where(labels == 0)[0]
        max_fg = int(self.train_rois_per_image * self.train_fg_fraction)
        n_fg = min(fg_idx.numel(), max_fg)
        n_bg = min(bg_idx.numel(), self.train_rois_per_image - n_fg)
        if n_fg > 0:
            fg_idx = fg_idx[torch.randperm(fg_idx.numel(), device=device)[:n_fg]]
        if n_bg > 0:
            bg_idx = bg_idx[torch.randperm(bg_idx.numel(), device=device)[:n_bg]]
        keep = torch.cat([fg_idx, bg_idx])

        s_boxes = all_boxes[keep]
        s_labels = labels[keep]
        s_gt_idx = matched_gt[keep]

        bbox_targets = encode_boxes(s_boxes, gt_boxes[s_gt_idx])
        bbox_targets[s_labels == 0] = 0

        # Mask targets: crop GT mask within RoI, resize to m×m (paper: "intersection of RoI and GT mask")
        mask_targets = torch.zeros(keep.numel(), ms, ms, device=device)
        for i in range(n_fg):
            gi = s_gt_idx[i]
            mf = gt_masks[gi].float()
            b = s_boxes[i]
            x1, y1 = int(b[0].clamp(min=0)), int(b[1].clamp(min=0))
            x2, y2 = int(b[2].clamp(min=1)), int(b[3].clamp(min=1))
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            if x2 > mf.size(1): x2 = mf.size(1)
            if y2 > mf.size(0): y2 = mf.size(0)
            if x1 >= x2 or y1 >= y2:
                continue
            crop = mf[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            mask_targets[i] = (F.interpolate(
                crop, (ms, ms), mode='bilinear', align_corners=False
            ).squeeze() > 0.5).float()

        return s_boxes, s_labels, bbox_targets, mask_targets, n_fg

    def forward(self, images, image_sizes, training=False,
                gt_boxes=None, gt_classes=None, gt_masks=None):
        B = images.size(0)
        device = images.device
        results: Dict[str, torch.Tensor] = {}

        features = self.backbone(images)

        proposals_list, rpn_loss = [], torch.tensor(0.0, device=device)
        for i, sz in enumerate(image_sizes):
            gt_b = gt_boxes[i] if training and gt_boxes else None
            p, l = self.rpn([features[i:i+1]], sz, gt_boxes=gt_b, training=training)
            proposals_list.append(p)
            if l is not None:
                rpn_loss = rpn_loss + l
        rpn_loss /= max(B, 1)
        results['rpn_loss'] = rpn_loss

        if training:
            cls_loss = bbox_loss = mask_loss = torch.tensor(0.0, device=device)
            for i in range(B):
                gb = gt_boxes[i].to(device)
                gc = gt_classes[i].to(device)
                gm = gt_masks[i].to(device)
                if gb.numel() == 0:
                    continue
                s_boxes, s_labels, b_tgt, m_tgt, n_fg = \
                    self._sample_proposals(proposals_list[i], gb, gc, gm, device)
                if s_boxes.numel() == 0:
                    continue
                roi_f = self.roi_align(features[i:i+1], s_boxes)
                cl, bd, ml = self.heads(roi_f)
                cls_loss = cls_loss + F.cross_entropy(cl, s_labels)

                fg = s_labels > 0
                if fg.sum() > 0:
                    fc = s_labels[fg]
                    fbp = bd[fg].view(-1, self.num_classes, 4)
                    idx = fc.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4)
                    bbox_loss = bbox_loss + F.smooth_l1_loss(
                        fbp.gather(1, idx).squeeze(1), b_tgt[fg])

                    # Mask loss on FG only using class-specific channel (paper Sec 3)
                    fml = ml[fg]
                    fmt = m_tgt[:fg.sum()]
                    ml_sum = torch.tensor(0.0, device=device)
                    for j in range(fg.sum()):
                        ci = min(int(fc[j].item()), fml.size(1) - 1)
                        ml_sum = ml_sum + F.binary_cross_entropy_with_logits(
                            fml[j, ci], fmt[j])
                    mask_loss = mask_loss + ml_sum / fg.sum()

            cls_loss /= B
            bbox_loss /= B
            mask_loss /= B
            total = rpn_loss + cls_loss + bbox_loss + mask_loss

            results.update({
                'classification_loss': cls_loss,
                'bbox_loss': bbox_loss,
                'mask_loss': mask_loss,
                'total_loss': total,
            })
            # For logging: run heads on a few proposals
            ap = torch.cat(proposals_list, dim=0)
            results['proposals'] = ap
            n = min(ap.size(0), 50)
            rf = self.roi_align(features, ap[:n])
            cl, bd, ml = self.heads(rf)
            results['class_logits'] = cl
            results['bbox_deltas'] = bd
            results['mask_logits'] = ml
            return results

        # ── Inference ──
        all_props = torch.cat(proposals_list, dim=0)
        results['proposals'] = all_props
        if all_props.size(0) == 0:
            z = lambda *s: torch.zeros(*s, device=device)
            results.update({'class_logits': z(0, self.num_classes),
                            'bbox_deltas': z(0, self.num_classes * 4),
                            'mask_logits': z(0, self.num_classes, self.mask_output_size, self.mask_output_size)})
            return results
        rf = self.roi_align(features, all_props)
        cl, bd, ml = self.heads(rf)
        results['class_logits'] = cl
        results['bbox_deltas'] = bd
        results['mask_logits'] = ml
        return results

    def inference(self, images, image_sizes, score_threshold=0.05, nms_threshold=0.5):
        """Paper inference: proposals → box branch → per-class NMS → top 100 → mask."""
        with torch.no_grad():
            out = self.forward(images, image_sizes, training=False)
        logits = out['class_logits']
        bds = out['bbox_deltas']
        masks = out['mask_logits']
        props = out['proposals']
        n = logits.size(0)
        if n == 0:
            return [{'class_ids': torch.tensor([], dtype=torch.long),
                     'scores': torch.tensor([]),
                     'boxes': torch.zeros(0, 4),
                     'masks': torch.zeros(0, self.num_classes, self.mask_output_size, self.mask_output_size)}]

        if props.size(0) > n:
            props = props[:n]

        probs = F.softmax(logits, dim=1)
        bd = bds.view(n, self.num_classes, 4)

        # Per-class NMS (paper: "non-maximum suppression" after box branch)
        all_boxes, all_scores, all_cids, all_mask_idx = [], [], [], []
        for c in range(1, self.num_classes):
            sc = probs[:, c]
            keep = sc > score_threshold
            if keep.sum() == 0:
                continue
            sc_k = sc[keep]
            deltas_c = bd[keep, c]
            boxes_c = decode_boxes(props[keep], deltas_c)
            # Clip to image
            boxes_c[:, 0].clamp_(min=0); boxes_c[:, 1].clamp_(min=0)
            boxes_c[:, 2].clamp_(max=images.size(3)); boxes_c[:, 3].clamp_(max=images.size(2))
            nms_keep = nms(boxes_c, sc_k, nms_threshold)
            all_boxes.append(boxes_c[nms_keep])
            all_scores.append(sc_k[nms_keep])
            all_cids.append(torch.full((nms_keep.numel(),), c, dtype=torch.long))
            # Track which original proposal index → for mask lookup
            orig_idx = torch.where(keep)[0][nms_keep]
            all_mask_idx.append(orig_idx)

        if not all_boxes:
            return [{'class_ids': torch.tensor([], dtype=torch.long),
                     'scores': torch.tensor([]),
                     'boxes': torch.zeros(0, 4),
                     'masks': torch.zeros(0, self.num_classes, self.mask_output_size, self.mask_output_size)}]

        final_boxes = torch.cat(all_boxes)
        final_scores = torch.cat(all_scores)
        final_cids = torch.cat(all_cids)
        final_midx = torch.cat(all_mask_idx)

        # Keep top 100 (paper: "highest scoring 100 detection boxes")
        if final_scores.numel() > 100:
            topk = final_scores.topk(100).indices
            final_boxes = final_boxes[topk]
            final_scores = final_scores[topk]
            final_cids = final_cids[topk]
            final_midx = final_midx[topk]

        # Mask: use k-th mask, binarize at 0.5 (paper: "k-th mask where k is predicted class")
        final_masks = masks[final_midx]
        final_masks = (torch.sigmoid(final_masks) > 0.5).float()

        return [{'class_ids': final_cids, 'scores': final_scores,
                 'boxes': final_boxes, 'masks': final_masks}]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
