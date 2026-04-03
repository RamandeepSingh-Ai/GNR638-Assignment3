"""Evaluate both models at multiple thresholds, find optimal, save metrics."""
import sys, torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, '.')

from dataset import create_dataloader
from compare import compute_box_iou, compute_mask_iou

device = torch.device('cpu')


def paste_mask_in_image(mask_pred, box, image_size=(128, 128)):
    """Resize mask from m×m to box size, paste into full image."""
    H, W = image_size
    x1 = int(max(0, box[0].item())); y1 = int(max(0, box[1].item()))
    x2 = int(min(W, box[2].item())); y2 = int(min(H, box[3].item()))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), dtype=np.float32)
    m = torch.tensor(mask_pred).float().unsqueeze(0).unsqueeze(0)
    m_r = F.interpolate(m, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    out = np.zeros((H, W), dtype=np.float32)
    out[y1:y2, x1:x2] = (m_r.squeeze().numpy() > 0.5).astype(np.float32)
    return out


def evaluate_at_threshold(model, threshold, num_samples=50, image_size=(128, 128)):
    model.eval()
    dl = create_dataloader(num_samples=num_samples, batch_size=1, seed=123)
    box_ious, mask_ious = [], []
    class_correct = class_total = total_preds = 0

    with torch.no_grad():
        for idx, batch in enumerate(dl):
            if idx >= num_samples: break
            imgs = batch['image'].to(device)
            results = model.inference(imgs, [image_size], score_threshold=threshold)

            for img_idx, result in enumerate(results):
                pb = result['boxes'].cpu()
                pm = result['masks'].cpu()
                pc = result['class_ids'].cpu()
                total_preds += pb.shape[0]
                gb = batch['boxes'][img_idx]
                gm = batch['masks'][img_idx]
                gl = batch['labels'][img_idx]

                for j in range(len(gl)):
                    if pb.shape[0] == 0:
                        box_ious.append(0.0); mask_ious.append(0.0)
                        class_total += 1; continue
                    ious = [compute_box_iou(gb[j], p) for p in pb]
                    bi = int(np.argmax(ious))
                    box_ious.append(ious[bi])
                    # Mask IoU: paste predicted mask into image at box location
                    if bi < pm.shape[0]:
                        ci = min(int(pc[bi].item()), pm.shape[1] - 1)
                        pred_full = paste_mask_in_image(pm[bi, ci].numpy(), pb[bi], image_size)
                        mask_ious.append(compute_mask_iou(gm[j].numpy(), pred_full))
                    else:
                        mask_ious.append(0.0)
                    if bi < pc.shape[0]:
                        if pc[bi].item() == gl[j].item():
                            class_correct += 1
                    class_total += 1

    return {
        'box_iou_mean': float(np.mean(box_ious)) if box_ious else 0.0,
        'mask_iou_mean': float(np.mean(mask_ious)) if mask_ious else 0.0,
        'class_accuracy': class_correct / max(1, class_total),
        'total_preds': total_preds,
        'num_gt': len(box_ious),
    }


def find_best(model, name, thresholds=(0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)):
    print(f"\n  Evaluating {name}...")
    best_t, best_box = 0.05, 0.0
    all_r = {}
    for t in thresholds:
        m = evaluate_at_threshold(model, t)
        all_r[t] = m
        print(f"    t={t:.2f}: box={m['box_iou_mean']*100:5.1f}%  "
              f"mask={m['mask_iou_mean']*100:5.1f}%  "
              f"cls={m['class_accuracy']*100:5.1f}%  "
              f"preds={m['total_preds']}")
        if m['box_iou_mean'] > best_box:
            best_box = m['box_iou_mean']; best_t = t
    print(f"  Best threshold: {best_t}")
    return best_t, all_r[best_t]


if __name__ == '__main__':
    from model import SimpleMaskRCNN
    from improved_model import ImprovedMaskRCNN

    print("="*65)
    print("  EVALUATION")
    print("="*65)

    base = SimpleMaskRCNN(num_classes=5)
    base.load_state_dict(torch.load('./checkpoints/baseline_best.pth', map_location='cpu', weights_only=True))
    bt, bm = find_best(base, 'Baseline (SimpleCNN)')
    torch.save(bm, './checkpoints/baseline_metrics.pt')

    imp = ImprovedMaskRCNN(num_classes=5, pretrained=False)
    imp.load_state_dict(torch.load('./checkpoints/improved_best.pth', map_location='cpu', weights_only=True))
    it, im = find_best(imp, 'Improved (ResNet-18)')
    torch.save(im, './checkpoints/improved_metrics.pt')

    paper = {'box_iou_mean': 0.588, 'mask_iou_mean': 0.571, 'class_accuracy': 0.85}
    print("\n" + "="*65)
    print(f"  {'Metric':<18} {'Baseline':>12} {'Improved':>12} {'Paper':>12}")
    print("  " + "-"*60)
    for lbl, k in [('Box IoU','box_iou_mean'), ('Mask IoU','mask_iou_mean'), ('Cls Acc','class_accuracy')]:
        print(f"  {lbl:<18} {bm[k]*100:>10.1f}%  {im[k]*100:>10.1f}%  {paper[k]*100:>10.1f}%")
    print("="*65)
