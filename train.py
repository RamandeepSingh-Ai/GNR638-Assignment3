"""
Training pipeline for SimpleMaskRCNN.

Includes training loop, validation, checkpointing, and LR scheduling.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

sys.path.insert(0, '.')


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 5,
) -> Dict[str, float]:
    model.train()

    totals = {k: 0.0 for k in ['total_loss', 'rpn_loss', 'classification_loss', 'bbox_loss', 'mask_loss']}
    num_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        boxes_list = [b.to(device) for b in batch['boxes']]
        labels_list = [lb.to(device) for lb in batch['labels']]
        masks_list = [m.to(device) for m in batch['masks']]

        image_size = (images.size(2), images.size(3))
        image_sizes = [image_size] * images.size(0)

        outputs = model(
            images, image_sizes, training=True,
            gt_boxes=boxes_list, gt_classes=labels_list, gt_masks=masks_list,
        )

        total_loss = outputs['total_loss']
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = images.size(0)
        num_samples += bs
        for key in totals:
            val = outputs.get(key, torch.tensor(0.0))
            totals[key] += val.item() * bs

        if (batch_idx + 1) % print_freq == 0:
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}]  Loss: {total_loss.item():.4f}")

    return {k: v / max(1, num_samples) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, num_samples = 0.0, 0

    for batch in dataloader:
        images = batch['image'].to(device)
        boxes_list = [b.to(device) for b in batch['boxes']]
        labels_list = [lb.to(device) for lb in batch['labels']]
        masks_list = [m.to(device) for m in batch['masks']]

        image_size = (images.size(2), images.size(3))
        image_sizes = [image_size] * images.size(0)

        outputs = model(
            images, image_sizes, training=True,
            gt_boxes=boxes_list, gt_classes=labels_list, gt_masks=masks_list,
        )
        total_loss += outputs['total_loss'].item() * images.size(0)
        num_samples += images.size(0)

    return {'val_loss': total_loss / max(1, num_samples)}


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints',
) -> Dict[str, list]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history: Dict[str, list] = {
        'train_loss': [], 'train_rpn_loss': [], 'train_cls_loss': [],
        'train_bbox_loss': [], 'train_mask_loss': [], 'val_loss': [],
    }
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        train_losses = train_one_epoch(model, train_dataloader, optimizer, device, epoch)

        history['train_loss'].append(train_losses['total_loss'])
        history['train_rpn_loss'].append(train_losses['rpn_loss'])
        history['train_cls_loss'].append(train_losses['classification_loss'])
        history['train_bbox_loss'].append(train_losses['bbox_loss'])
        history['train_mask_loss'].append(train_losses['mask_loss'])

        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"    RPN: {train_losses['rpn_loss']:.4f}  "
              f"Cls: {train_losses['classification_loss']:.4f}  "
              f"BBox: {train_losses['bbox_loss']:.4f}  "
              f"Mask: {train_losses['mask_loss']:.4f}")

        if val_dataloader is not None:
            val_metrics = validate(model, val_dataloader, device)
            history['val_loss'].append(val_metrics['val_loss'])
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")

            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        if epoch % 5 == 0:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), path)
            print(f"  -> Saved checkpoint: {path}")

        scheduler.step()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history


def main():
    from dataset import create_dataloader
    from model import SimpleMaskRCNN

    print("SimpleMaskRCNN Training")
    print("=" * 50)

    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001

    train_loader = create_dataloader(num_samples=80, batch_size=BATCH_SIZE, seed=42)
    val_loader = create_dataloader(num_samples=20, batch_size=BATCH_SIZE, seed=99)

    model = SimpleMaskRCNN(num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = train(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        device=device, checkpoint_dir='./checkpoints',
    )

    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss:   {history['val_loss'][-1]:.4f}")


if __name__ == '__main__':
    main()
