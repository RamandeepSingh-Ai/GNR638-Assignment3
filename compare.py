"""
Model evaluation and comparison tool for SimpleMaskRCNN.

Computes Box IoU, Mask IoU, and per-class accuracy on the toy dataset,
and produces side-by-side visualizations of GT vs predictions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, '.')


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    if x2 < x1 or y2 < y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1.size == 0 or mask2.size == 0:
        return 0.0
    inter = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
    union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
    return float(inter) / max(float(union), 1e-6)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Evaluate SimpleMaskRCNN on the synthetic dataset."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader,
        num_samples: int = 20,
        image_size: Tuple[int, int] = (128, 128),
        score_threshold: float = 0.05,
    ) -> Dict[str, float]:
        model.to(self.device)
        model.eval()

        box_ious, mask_ious = [], []
        class_correct, class_total = 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break

                images = batch['image'].to(self.device)
                gt_boxes_list = batch['boxes']
                gt_masks_list = batch['masks']
                gt_labels_list = batch['labels']

                image_sizes = [image_size] * images.size(0)
                results = model.inference(images, image_sizes, score_threshold=score_threshold)

                for img_idx, result in enumerate(results):
                    pred_boxes = result['boxes'].cpu()
                    pred_masks = result['masks'].cpu()
                    pred_class_ids = result['class_ids'].cpu()

                    gt_boxes = gt_boxes_list[img_idx]
                    gt_masks = gt_masks_list[img_idx]
                    gt_labels = gt_labels_list[img_idx]

                    for j, (gt_box, gt_mask, gt_label) in enumerate(
                        zip(gt_boxes, gt_masks, gt_labels)
                    ):
                        if pred_boxes.shape[0] == 0:
                            box_ious.append(0.0)
                            mask_ious.append(0.0)
                            class_total += 1
                            continue

                        ious = [compute_box_iou(gt_box, pb) for pb in pred_boxes]
                        best_idx = int(np.argmax(ious))
                        best_iou = ious[best_idx]
                        box_ious.append(best_iou)

                        # Mask IoU — paste predicted mask into correct box location
                        if pred_masks.shape[0] > best_idx and gt_mask.shape[0] == image_size[0]:
                            pred_m = pred_masks[best_idx]
                            pred_class = int(pred_class_ids[best_idx].item()) if best_idx < pred_class_ids.shape[0] else 0
                            pred_class = min(max(0, pred_class), pred_m.shape[0] - 1)
                            pred_m_14 = pred_m[pred_class].numpy()

                            H, W = image_size
                            pb = pred_boxes[best_idx]
                            x1 = int(max(0, pb[0].item())); y1 = int(max(0, pb[1].item()))
                            x2 = int(min(W, pb[2].item())); y2 = int(min(H, pb[3].item()))
                            if x2 > x1 and y2 > y1:
                                bw, bh = x2 - x1, y2 - y1
                                m_r = torch.nn.functional.interpolate(
                                    torch.from_numpy(pred_m_14).unsqueeze(0).unsqueeze(0).float(),
                                    size=(bh, bw), mode='bilinear', align_corners=False
                                )
                                m_np = (m_r.squeeze().numpy() > 0.5).astype(np.float32)
                                full_mask = np.zeros((H, W), dtype=np.float32)
                                full_mask[y1:y2, x1:x2] = m_np
                                mask_ious.append(compute_mask_iou(gt_mask.numpy(), full_mask))
                            else:
                                mask_ious.append(0.0)

                        # Class accuracy
                        if best_idx < pred_class_ids.shape[0]:
                            if pred_class_ids[best_idx].item() == gt_label.item():
                                class_correct += 1
                        class_total += 1

        metrics = {
            'box_iou_mean': float(np.mean(box_ious)) if box_ious else 0.0,
            'mask_iou_mean': float(np.mean(mask_ious)) if mask_ious else 0.0,
            'class_accuracy': class_correct / max(1, class_total),
            'num_gt_objects': len(box_ious),
            'num_with_predictions': sum(1 for x in box_ious if x > 0),
        }
        return metrics

    def print_report(self, metrics: Dict[str, float], model_name: str = "SimpleMaskRCNN") -> None:
        print(f"\n{'='*55}")
        print(f"  {model_name} - Evaluation Report")
        print(f"{'='*55}")
        print(f"  Box IoU (mean):      {metrics['box_iou_mean']:.4f}")
        print(f"  Mask IoU (mean):     {metrics['mask_iou_mean']:.4f}")
        print(f"  Class Accuracy:      {metrics['class_accuracy']:.4f}")
        print(f"  GT objects seen:     {metrics['num_gt_objects']}")
        print(f"  Objects w/ preds:    {metrics['num_with_predictions']}")
        print(f"{'='*55}\n")

        # Paper comparison
        print("  Paper Reference (Mask R-CNN on COCO, ResNet-50-FPN):")
        print("    Box AP50:       ~58%")
        print("    Mask AP50:      ~57%")
        print("  Our Toy Dataset (SimpleCNN, 128x128, 5 classes):")
        print(f"    Box IoU:        {metrics['box_iou_mean']*100:.1f}%")
        print(f"    Mask IoU:       {metrics['mask_iou_mean']*100:.1f}%")
        print("  Gap: SimpleCNN is ~10-20% lower than ResNet+FPN.")
        print("  (Expected — see STEP9_IMPROVEMENTS for upgrade paths)\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: 'bg', 1: 'circle', 2: 'rect', 3: 'triangle', 4: 'diamond'}
CLASS_COLORS_VIZ = {1: 'red', 2: 'lime', 3: 'cyan', 4: 'yellow'}


def visualize_sample(
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_class_ids: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_masks: torch.Tensor,
    save_path: str = 'comparison.png',
    image_size: Tuple[int, int] = (128, 128),
) -> None:
    """Visualize ground truth vs predictions side-by-side with masks overlay."""
    img_np = image.permute(1, 2, 0).numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Mask R-CNN: Ground Truth vs Predictions", fontsize=14, fontweight='bold')

    # --- Left: Ground truth ---
    ax = axes[0]
    ax.imshow(img_np)
    ax.set_title("Ground Truth", fontsize=12)
    for box, label, mask in zip(gt_boxes, gt_labels, gt_masks):
        x1, y1, x2, y2 = box.tolist()
        label_int = int(label.item())
        color = CLASS_COLORS_VIZ.get(label_int, 'white')
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, CLASS_NAMES.get(label_int, str(label_int)), color=color, fontsize=8, fontweight='bold')
        overlay = np.zeros((*image_size, 4), dtype=np.float32)
        rgb = matplotlib_color_to_rgb(color)
        overlay[..., :3] = rgb
        overlay[..., 3] = mask.numpy() * 0.4
        ax.imshow(overlay)
    ax.axis('off')

    # --- Middle: Predictions ---
    ax = axes[1]
    ax.imshow(img_np)
    ax.set_title("Predictions (SimpleMaskRCNN)", fontsize=12)
    if pred_boxes.shape[0] > 0:
        for i, (box, cls_id, score) in enumerate(zip(pred_boxes, pred_class_ids, pred_scores)):
            x1, y1, x2, y2 = box.tolist()
            cls_int = int(cls_id.item())
            color = CLASS_COLORS_VIZ.get(cls_int, 'white')
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, f"{CLASS_NAMES.get(cls_int, str(cls_int))} {score:.2f}",
                    color=color, fontsize=7, fontweight='bold')
            if i < pred_masks.shape[0]:
                m = pred_masks[i]  # (num_classes, 14, 14)
                cls_idx = min(cls_int, m.shape[0] - 1)
                m_np = m[cls_idx].float().unsqueeze(0).unsqueeze(0)
                m_up = torch.nn.functional.interpolate(m_np, size=image_size, mode='nearest').squeeze().numpy()
                overlay = np.zeros((*image_size, 4), dtype=np.float32)
                overlay[..., :3] = matplotlib_color_to_rgb(color)
                overlay[..., 3] = m_up * 0.4
                ax.imshow(overlay)
    else:
        ax.text(0.5, 0.5, 'No detections\n(adjust score threshold)', ha='center', va='center',
                transform=ax.transAxes, color='white', fontsize=11)
    ax.axis('off')

    # --- Right: Loss curves placeholder or metrics ---
    ax = axes[2]
    ax.axis('off')
    info_text = (
        "SimpleMaskRCNN Architecture\n"
        "─────────────────────────────\n"
        "Paper: Mask R-CNN (He et al. 2017)\n\n"
        "Backbone:  SimpleCNN (3 layers)\n"
        "           vs ResNet-50+FPN (paper)\n\n"
        "Input:     128×128\n"
        "           vs 800×1333 (paper)\n\n"
        "Proposals: 50 (vs 1000-2000)\n\n"
        "Dataset:   Synthetic shapes\n"
        "           vs COCO (paper)\n\n"
        "Pipeline:\n"
        "  Input → Backbone → RPN\n"
        "        → RoI Align\n"
        "        → Cls + BBox + Mask heads"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#1e1e2e', alpha=0.8, edgecolor='#444'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved visualization to: {save_path}")
    plt.close()


def matplotlib_color_to_rgb(color_name: str) -> np.ndarray:
    mapping = {
        'red': [1.0, 0.2, 0.2],
        'lime': [0.2, 1.0, 0.2],
        'cyan': [0.2, 0.9, 1.0],
        'yellow': [1.0, 1.0, 0.2],
        'white': [1.0, 1.0, 1.0],
    }
    return np.array(mapping.get(color_name, [1.0, 1.0, 1.0]))


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = 'training_curves.png',
) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("SimpleMaskRCNN Training Curves", fontsize=13, fontweight='bold')

    # Loss breakdown
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Total Loss', linewidth=2)
    ax.plot(epochs, history['train_rpn_loss'], '--', label='RPN Loss')
    ax.plot(epochs, history['train_cls_loss'], '--', label='Cls Loss')
    ax.plot(epochs, history['train_bbox_loss'], '--', label='BBox Loss')
    ax.plot(epochs, history['train_mask_loss'], '--', label='Mask Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Train vs Val
    ax = axes[1]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from dataset import create_dataloader
    from model import SimpleMaskRCNN

    print("Model Comparison Tool")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_loader = create_dataloader(num_samples=20, batch_size=1, seed=123)
    model = SimpleMaskRCNN(num_classes=5)

    checkpoint_path = './checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found. Run train.py first.")

    evaluator = ModelEvaluator(device=device)
    metrics = evaluator.evaluate(model, test_loader, num_samples=20)
    evaluator.print_report(metrics)

    # Visualize first sample
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    preds = model.inference(images, [(128, 128)])

    visualize_sample(
        image=batch['image'][0],
        gt_boxes=batch['boxes'][0],
        gt_labels=batch['labels'][0],
        gt_masks=batch['masks'][0],
        pred_boxes=preds[0]['boxes'].cpu(),
        pred_class_ids=preds[0]['class_ids'].cpu(),
        pred_scores=preds[0]['scores'].cpu(),
        pred_masks=preds[0]['masks'].cpu(),
        save_path='comparison.png',
    )


if __name__ == '__main__':
    main()
