"""
Full paper-faithful experiment.
  - 300 train / 75 val synthetic shapes
  - 30 epochs each model
  - Paper training: GT injection, IoU matching, balanced sampling, class-specific masks
  - Paper inference: per-class NMS, top-100, box decoding
"""
import sys, os, time, random
import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, '.')

from dataset import ShapeDataset, custom_collate_fn
from torch.utils.data import DataLoader

torch.manual_seed(42); np.random.seed(42); random.seed(42)


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, base, augment=True):
        self.base = base; self.augment = augment
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]
        if not self.augment: return s
        img, boxes, labels, masks = s['image'], s['boxes'].clone(), s['labels'], s['masks'].clone()
        H, W = img.shape[1], img.shape[2]
        if random.random() > 0.5:
            img = torch.flip(img, [2]); masks = torch.flip(masks, [2])
            if boxes.shape[0] > 0:
                x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0] = W - x2; boxes[:, 2] = W - x1
        img = (img * random.uniform(0.8, 1.2)).clamp(0, 1)
        return {'image': img, 'boxes': boxes, 'labels': labels, 'masks': masks}


def train_one(model, tl, vl, epochs, lr, device, name, backbone_lr=False):
    os.makedirs('./checkpoints', exist_ok=True)
    model.to(device)

    if backbone_lr:
        bb = list(model.backbone.parameters())
        bb_ids = set(id(p) for p in bb)
        other = [p for p in model.parameters() if id(p) not in bb_ids]
        opt = optim.AdamW([{'params': bb, 'lr': lr*0.1}, {'params': other, 'lr': lr}], weight_decay=1e-4)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    hist = {k: [] for k in ['train_loss','train_rpn','train_cls','train_bbox','train_mask','val_loss']}
    best = float('inf')

    for ep in range(1, epochs+1):
        model.train()
        tots = {k: 0.0 for k in ['total_loss','rpn_loss','classification_loss','bbox_loss','mask_loss']}
        n = 0
        for batch in tl:
            imgs = batch['image'].to(device)
            bs = imgs.size(0); sz = [(128,128)] * bs
            out = model(imgs, sz, training=True,
                        gt_boxes=[b.to(device) for b in batch['boxes']],
                        gt_classes=[l.to(device) for l in batch['labels']],
                        gt_masks=[m.to(device) for m in batch['masks']])
            loss = out['total_loss']
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += bs
            for k in tots: tots[k] += out.get(k, torch.tensor(0.0)).item() * bs
        sched.step()
        for k in tots: tots[k] /= max(1, n)
        hist['train_loss'].append(tots['total_loss'])
        hist['train_rpn'].append(tots['rpn_loss'])
        hist['train_cls'].append(tots['classification_loss'])
        hist['train_bbox'].append(tots['bbox_loss'])
        hist['train_mask'].append(tots['mask_loss'])

        model.eval()
        vl_sum, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                imgs = batch['image'].to(device); bs = imgs.size(0)
                out = model(imgs, [(128,128)]*bs, training=True,
                            gt_boxes=[b.to(device) for b in batch['boxes']],
                            gt_classes=[l.to(device) for l in batch['labels']],
                            gt_masks=[m.to(device) for m in batch['masks']])
                vl_sum += out['total_loss'].item() * bs; vn += bs
        vl_avg = vl_sum / max(1, vn)
        hist['val_loss'].append(vl_avg)

        print(f"  [{name}] Ep {ep:2d}/{epochs} | "
              f"train={tots['total_loss']:.4f} (rpn={tots['rpn_loss']:.3f} "
              f"cls={tots['classification_loss']:.3f} bbox={tots['bbox_loss']:.3f} "
              f"mask={tots['mask_loss']:.3f}) | val={vl_avg:.4f}")

        if vl_avg < best:
            best = vl_avg
            torch.save(model.state_dict(), f'./checkpoints/{name}_best.pth')
        if ep % 10 == 0:
            torch.save(model.state_dict(), f'./checkpoints/{name}_ep{ep}.pth')

    torch.save(hist, f'./checkpoints/{name}_history.pt')
    return hist


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Creating datasets...")
    train_ds = ShapeDataset(num_samples=300, seed=42)
    val_ds   = ShapeDataset(num_samples=75,  seed=99)
    tl = DataLoader(AugDataset(train_ds, True),  batch_size=4, shuffle=True,  num_workers=0, collate_fn=custom_collate_fn)
    vl = DataLoader(AugDataset(val_ds,   False), batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    print(f"Train: {len(train_ds)} ({len(tl)} batches)  Val: {len(val_ds)} ({len(vl)} batches)")

    # ── Baseline ──
    print("\n" + "="*70)
    print("  BASELINE: SimpleCNN Mask R-CNN (paper-faithful, 30 epochs)")
    print("="*70)
    from model import SimpleMaskRCNN
    base = SimpleMaskRCNN(num_classes=5)
    print(f"  Params: {base.count_parameters():,}")
    t0 = time.time()
    bh = train_one(base, tl, vl, epochs=30, lr=0.001, device=device, name='baseline')
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── Improved ──
    print("\n" + "="*70)
    print("  IMPROVED: ResNet-18 Mask R-CNN (paper-faithful, 30 epochs)")
    print("="*70)
    from improved_model import ImprovedMaskRCNN
    imp = ImprovedMaskRCNN(num_classes=5, pretrained=True)
    print(f"  Params: {imp.count_parameters():,}")
    t0 = time.time()
    ih = train_one(imp, tl, vl, epochs=30, lr=0.001, device=device, name='improved', backbone_lr=True)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("\n  Training complete! Run evaluate_final.py and generate_report.py next.")
