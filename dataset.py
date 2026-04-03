"""
Synthetic shape dataset for Mask R-CNN training.

Generates random 128x128 images containing geometric shapes (circles,
rectangles, triangles, diamonds) with ground truth bounding boxes and
instance segmentation masks.
"""

import random
import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, List, Dict


class ShapeDataset(data.Dataset):
    """
    Synthetic dataset of geometric shapes with instance masks.

    Classes:
        0: background
        1: circle
        2: rectangle
        3: triangle
        4: diamond
    """

    SHAPE_CLASSES = {0: 'background', 1: 'circle', 2: 'rectangle', 3: 'triangle', 4: 'diamond'}
    CLASS_COLORS = {1: (220, 80, 80), 2: (80, 220, 80), 3: (80, 80, 220), 4: (220, 220, 80)}

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (128, 128),
        max_shapes_per_image: int = 3,
        seed: int = 42,
    ) -> None:
        super(ShapeDataset, self).__init__()

        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        self.num_samples = num_samples
        self.image_size = image_size
        self.max_shapes_per_image = max_shapes_per_image

        # Pre-generate all samples for reproducibility
        rng = np.random.RandomState(seed)
        self._samples = [self._generate_image(rng) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, boxes, labels, masks = self._samples[idx]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.from_numpy(boxes).float() if boxes.shape[0] > 0 else torch.zeros((0, 4))
        labels_t = torch.from_numpy(labels).long() if labels.shape[0] > 0 else torch.zeros(0, dtype=torch.long)
        masks_t = torch.from_numpy(masks).float() if masks.shape[0] > 0 else torch.zeros((0, *self.image_size))

        return {'image': image_t, 'boxes': boxes_t, 'labels': labels_t, 'masks': masks_t}

    def _generate_image(self, rng: np.random.RandomState):
        h, w = self.image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)

        num_shapes = rng.randint(1, self.max_shapes_per_image + 1)
        boxes_list, labels_list, masks_list = [], [], []

        for _ in range(num_shapes):
            shape_type = rng.randint(1, 5)  # 1-4
            base_color = self.CLASS_COLORS[shape_type]
            jitter = rng.randint(-30, 30, 3)
            color = tuple(int(np.clip(c + j, 50, 255)) for c, j in zip(base_color, jitter))

            if shape_type == 1:
                box, mask = self._draw_circle(image, color, rng)
            elif shape_type == 2:
                box, mask = self._draw_rectangle(image, color, rng)
            elif shape_type == 3:
                box, mask = self._draw_triangle(image, color, rng)
            else:
                box, mask = self._draw_diamond(image, color, rng)

            if mask.sum() > 0:
                boxes_list.append(box)
                labels_list.append(shape_type)
                masks_list.append(mask)

        if not boxes_list:
            boxes_list.append([10, 10, 30, 30])
            labels_list.append(2)
            m = np.zeros((h, w), dtype=np.float32)
            m[10:30, 10:30] = 1.0
            masks_list.append(m)

        return (
            image,
            np.array(boxes_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int64),
            np.array(masks_list, dtype=np.float32),
        )

    def _draw_circle(self, image, color, rng):
        h, w = image.shape[:2]
        radius = rng.randint(10, max(11, int(min(h, w) * 0.25)))
        cx = rng.randint(radius, w - radius)
        cy = rng.randint(radius, h - radius)

        mask = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.ogrid[:h, :w]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        image[circle] = color
        mask[circle] = 1.0

        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w - 1, cx + radius), min(h - 1, cy + radius)
        return np.array([x1, y1, x2, y2], dtype=np.float32), mask

    def _draw_rectangle(self, image, color, rng):
        h, w = image.shape[:2]
        rect_h = rng.randint(15, max(16, int(h * 0.45)))
        rect_w = rng.randint(15, max(16, int(w * 0.45)))
        x1 = rng.randint(0, w - rect_w)
        y1 = rng.randint(0, h - rect_h)
        x2 = min(w - 1, x1 + rect_w)
        y2 = min(h - 1, y1 + rect_h)

        mask = np.zeros((h, w), dtype=np.float32)
        image[y1:y2 + 1, x1:x2 + 1] = color
        mask[y1:y2 + 1, x1:x2 + 1] = 1.0
        return np.array([x1, y1, x2, y2], dtype=np.float32), mask

    def _draw_triangle(self, image, color, rng):
        h, w = image.shape[:2]
        size = rng.randint(15, max(16, int(min(h, w) * 0.35)))
        cx = rng.randint(size, w - size)
        cy = rng.randint(size, h - size)

        p1 = (cx, cy - size)
        p2 = (cx - size, cy + size)
        p3 = (cx + size, cy + size)

        mask = np.zeros((h, w), dtype=np.float32)
        min_x = max(0, min(p1[0], p2[0], p3[0]))
        max_x = min(w - 1, max(p1[0], p2[0], p3[0]))
        min_y = max(0, min(p1[1], p2[1], p3[1]))
        max_y = min(h - 1, max(p1[1], p2[1], p3[1]))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if self._point_in_triangle(x, y, p1, p2, p3):
                    image[y, x] = color
                    mask[y, x] = 1.0

        return np.array([min_x, min_y, max_x, max_y], dtype=np.float32), mask

    def _draw_diamond(self, image, color, rng):
        h, w = image.shape[:2]
        size = rng.randint(15, max(16, int(min(h, w) * 0.35)))
        cx = rng.randint(size, w - size)
        cy = rng.randint(size, h - size)

        mask = np.zeros((h, w), dtype=np.float32)
        min_x = max(0, cx - size)
        max_x = min(w - 1, cx + size)
        min_y = max(0, cy - size)
        max_y = min(h - 1, cy + size)

        yy, xx = np.ogrid[min_y:max_y + 1, min_x:max_x + 1]
        diamond = np.abs(xx - cx) + np.abs(yy - cy) <= size
        image[min_y:max_y + 1, min_x:max_x + 1][diamond] = color
        mask[min_y:max_y + 1, min_x:max_x + 1][diamond] = 1.0

        return np.array([min_x, min_y, max_x, max_y], dtype=np.float32), mask

    @staticmethod
    def _point_in_triangle(x, y, p1, p2, p3):
        def sign(px, py, ax, ay, bx, by):
            return (px - bx) * (ay - by) - (ax - bx) * (py - by)
        d1 = sign(x, y, p1[0], p1[1], p2[0], p2[1])
        d2 = sign(x, y, p2[0], p2[1], p3[0], p3[1])
        d3 = sign(x, y, p3[0], p3[1], p1[0], p1[1])
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)


def custom_collate_fn(batch):
    """Collate images into tensors; keep variable-length lists for boxes/labels/masks."""
    images = torch.stack([item['image'] for item in batch])
    return {
        'image': images,
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'masks': [item['masks'] for item in batch],
    }


def create_dataloader(
    num_samples: int = 100,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> data.DataLoader:
    dataset = ShapeDataset(num_samples=num_samples, seed=seed)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
