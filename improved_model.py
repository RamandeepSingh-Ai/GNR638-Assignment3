"""
Improved Mask R-CNN with ResNet-18 pretrained backbone.

Same paper-faithful training/inference as SimpleMaskRCNN, plus:
  - ResNet-18 pretrained (ImageNet)
  - Paper mask head (4 conv + deconv + 1×1)
  - Differential learning rate for backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from roi_align import ROIAlign
from rpn import RPN
from heads import DetectionHeads
from model import box_iou, nms, encode_boxes, decode_boxes


class ResNet18Backbone(nn.Module):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # stride 4, 64ch
        self.layer2 = resnet.layer2  # stride 8, 128ch
        self.proj = nn.Sequential(
            nn.Conv2d(128, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels
        self.stride = 8
        self.register_buffer('mean', self.MEAN)
        self.register_buffer('std',  self.STD)

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.proj(self.layer2(self.layer1(self.stem(x))))


class ImprovedMaskRCNN(nn.Module):
    """Mask R-CNN with ResNet-18 backbone — paper-faithful pipeline."""

    def __init__(self, num_classes=5, roi_pool_size=7, mask_output_size=28,
                 pretrained=True, rpn_post_nms=300,
                 train_rois=128, fg_frac=0.25, fg_iou=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.roi_pool_size = roi_pool_size
        self.mask_output_size = mask_output_size
        self.train_rois = train_rois
        self.fg_frac = fg_frac
        self.fg_iou = fg_iou

        self.backbone = ResNet18Backbone(pretrained=pretrained, out_channels=256)
        stride = self.backbone.stride
        self.rpn = RPN(
            in_channels=256, num_anchors=15,
            anchor_scales=(4, 8, 16, 32, 64), anchor_ratios=(0.5, 1.0, 2.0),
            strides=(stride,),
            pre_nms_top_k=2000, post_nms_top_k=rpn_post_nms, nms_thresh=0.7,
        )
        self.roi_align = ROIAlign(output_size=roi_pool_size, spatial_scale=1.0 / stride)
        self.heads = DetectionHeads(
            in_channels=256, pool_size=roi_pool_size,
            num_classes=num_classes, mask_output_size=mask_output_size,
        )

    def _sample_proposals(self, proposals, gt_boxes, gt_classes, gt_masks, device):
        ms = self.mask_output_size
        all_b = torch.cat([proposals, gt_boxes], dim=0)
        iou = box_iou(all_b, gt_boxes)
        max_iou, mg = iou.max(dim=1)
        labels = gt_classes[mg].long().to(device)
        labels[max_iou < self.fg_iou] = 0

        fg = torch.where(labels > 0)[0]
        bg = torch.where(labels == 0)[0]
        mfg = int(self.train_rois * self.fg_frac)
        nf = min(fg.numel(), mfg)
        nb = min(bg.numel(), self.train_rois - nf)
        if nf > 0: fg = fg[torch.randperm(fg.numel(), device=device)[:nf]]
        if nb > 0: bg = bg[torch.randperm(bg.numel(), device=device)[:nb]]
        keep = torch.cat([fg, bg])
        sb = all_b[keep]; sl = labels[keep]; sm = mg[keep]
        bt = encode_boxes(sb, gt_boxes[sm])
        bt[sl == 0] = 0

        mt = torch.zeros(keep.numel(), ms, ms, device=device)
        for i in range(nf):
            mf = gt_masks[sm[i]].float()
            b = sb[i]
            x1, y1 = int(b[0].clamp(min=0)), int(b[1].clamp(min=0))
            x2, y2 = int(b[2].clamp(min=1)), int(b[3].clamp(min=1))
            x2 = max(x2, x1+1); y2 = max(y2, y1+1)
            if x2 > mf.size(1): x2 = mf.size(1)
            if y2 > mf.size(0): y2 = mf.size(0)
            if x1 >= x2 or y1 >= y2: continue
            crop = mf[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            mt[i] = (F.interpolate(crop, (ms, ms), mode='bilinear', align_corners=False).squeeze() > 0.5).float()

        return sb, sl, bt, mt, nf

    def forward(self, images, image_sizes, training=False,
                gt_boxes=None, gt_classes=None, gt_masks=None):
        B = images.size(0); dev = images.device
        results: Dict[str, torch.Tensor] = {}
        features = self.backbone(images)

        pl, rpn_loss = [], torch.tensor(0.0, device=dev)
        for i, sz in enumerate(image_sizes):
            gb = gt_boxes[i] if training and gt_boxes else None
            p, l = self.rpn([features[i:i+1]], sz, gt_boxes=gb, training=training)
            pl.append(p)
            if l is not None: rpn_loss = rpn_loss + l
        rpn_loss /= max(B, 1)
        results['rpn_loss'] = rpn_loss

        if training:
            tcl = tbb = tmk = torch.tensor(0.0, device=dev)
            for i in range(B):
                gb = gt_boxes[i].to(dev); gc = gt_classes[i].to(dev); gm = gt_masks[i].to(dev)
                if gb.numel() == 0: continue
                sb, sl, bt, mt, nf = self._sample_proposals(pl[i], gb, gc, gm, dev)
                if sb.numel() == 0: continue
                rf = self.roi_align(features[i:i+1], sb)
                cl, bd, ml = self.heads(rf)
                tcl = tcl + F.cross_entropy(cl, sl)
                fg = sl > 0
                if fg.sum() > 0:
                    fc = sl[fg]
                    fbp = bd[fg].view(-1, self.num_classes, 4)
                    idx = fc.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4)
                    tbb = tbb + F.smooth_l1_loss(fbp.gather(1, idx).squeeze(1), bt[fg])
                    fml = ml[fg]; fmt = mt[:fg.sum()]
                    mls = torch.tensor(0.0, device=dev)
                    for j in range(fg.sum()):
                        ci = min(int(fc[j].item()), fml.size(1)-1)
                        mls = mls + F.binary_cross_entropy_with_logits(fml[j, ci], fmt[j])
                    tmk = tmk + mls / fg.sum()
            tcl /= B; tbb /= B; tmk /= B
            total = rpn_loss + tcl + tbb + tmk
            results.update({'classification_loss': tcl, 'bbox_loss': tbb,
                            'mask_loss': tmk, 'total_loss': total})
            ap = torch.cat(pl); results['proposals'] = ap
            n = min(ap.size(0), 50); rf = self.roi_align(features, ap[:n])
            cl, bd, ml = self.heads(rf)
            results['class_logits'] = cl; results['bbox_deltas'] = bd; results['mask_logits'] = ml
            return results

        ap = torch.cat(pl); results['proposals'] = ap
        if ap.size(0) == 0:
            z = lambda *s: torch.zeros(*s, device=dev)
            results.update({'class_logits': z(0, self.num_classes),
                            'bbox_deltas': z(0, self.num_classes*4),
                            'mask_logits': z(0, self.num_classes, self.mask_output_size, self.mask_output_size)})
            return results
        rf = self.roi_align(features, ap)
        cl, bd, ml = self.heads(rf)
        results['class_logits'] = cl; results['bbox_deltas'] = bd; results['mask_logits'] = ml
        return results

    def inference(self, images, image_sizes, score_threshold=0.05, nms_threshold=0.5):
        with torch.no_grad():
            out = self.forward(images, image_sizes, training=False)
        logits = out['class_logits']; bds = out['bbox_deltas']
        masks = out['mask_logits']; props = out['proposals']
        n = logits.size(0)
        if n == 0:
            return [{'class_ids': torch.tensor([], dtype=torch.long), 'scores': torch.tensor([]),
                     'boxes': torch.zeros(0, 4),
                     'masks': torch.zeros(0, self.num_classes, self.mask_output_size, self.mask_output_size)}]
        if props.size(0) > n: props = props[:n]
        probs = F.softmax(logits, dim=1)
        bd = bds.view(n, self.num_classes, 4)

        ab, asc, acid, amid = [], [], [], []
        for c in range(1, self.num_classes):
            sc = probs[:, c]
            keep = sc > score_threshold
            if keep.sum() == 0: continue
            sk = sc[keep]; dc = bd[keep, c]
            bc = decode_boxes(props[keep], dc)
            bc[:, 0].clamp_(min=0); bc[:, 1].clamp_(min=0)
            bc[:, 2].clamp_(max=images.size(3)); bc[:, 3].clamp_(max=images.size(2))
            nk = nms(bc, sk, nms_threshold)
            ab.append(bc[nk]); asc.append(sk[nk])
            acid.append(torch.full((nk.numel(),), c, dtype=torch.long))
            amid.append(torch.where(keep)[0][nk])

        if not ab:
            return [{'class_ids': torch.tensor([], dtype=torch.long), 'scores': torch.tensor([]),
                     'boxes': torch.zeros(0, 4),
                     'masks': torch.zeros(0, self.num_classes, self.mask_output_size, self.mask_output_size)}]
        fb = torch.cat(ab); fs = torch.cat(asc); fc = torch.cat(acid); fm = torch.cat(amid)
        if fs.numel() > 100:
            tk = fs.topk(100).indices
            fb, fs, fc, fm = fb[tk], fs[tk], fc[tk], fm[tk]
        fmasks = (torch.sigmoid(masks[fm]) > 0.5).float()
        return [{'class_ids': fc, 'scores': fs, 'boxes': fb, 'masks': fmasks}]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
