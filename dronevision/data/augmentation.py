"""
Data augmentation pipeline for DroneVision.

All augmentations are implemented from scratch using OpenCV and NumPy.
No external augmentation library (e.g. Albumentations) is required.

Every augmentation preserves bounding box coordinates and filters out
boxes that become degenerate (w < min_box_size or h < min_box_size)
after the transform.

Key augmentations for small drone detection:
  - Mosaic:      4-image mosaic forces model to detect drones in cluttered scenes.
  - MixUp:       Regularises feature representations.
  - RandomScale: Simulates altitude variation (drone → smaller/larger appearance).
  - ColorJitter: Handles diverse lighting and atmospheric conditions.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

_MIN_BOX_SIZE: float = 2.0   # pixels — boxes smaller than this are dropped
_MAX_JITTER: float = 0.1      # max random shift for mosaic center


def _clip_and_filter_boxes(
    boxes: np.ndarray,
    img_h: int,
    img_w: int,
    min_size: float = _MIN_BOX_SIZE,
) -> np.ndarray:
    """
    Clip normalized cxcywh boxes to [0,1] and remove degenerate ones.

    Args:
        boxes:    (N, 5) [cls, cx, cy, w, h] normalized.
        img_h, img_w: Current image dimensions (used for pixel size check).
        min_size: Minimum width and height in pixels.

    Returns:
        Filtered (M, 5) boxes where M <= N.
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy()
    # Clip to [0,1]
    boxes[:, 1:] = np.clip(boxes[:, 1:], 0.0, 1.0)

    # Convert to xyxy pixel for size check
    cx = boxes[:, 1] * img_w
    cy = boxes[:, 2] * img_h
    w = boxes[:, 3] * img_w
    h = boxes[:, 4] * img_h

    keep = (w >= min_size) & (h >= min_size)
    return boxes[keep]


class DroneAugmentation:
    """
    Configurable augmentation pipeline for drone detection training.

    Args:
        cfg: Augmentation config dict from the YAML config file.
             Expected keys (all optional, with defaults):
               horizontal_flip: float [0.5]
               vertical_flip:   float [0.3]
               color_jitter:    float [0.8]
               mosaic:          float [0.5]
               mixup:           float [0.1]
               random_scale:    float [0.5]
               gaussian_blur:   float [0.2]
        img_size: Target image size after augmentation.
    """

    def __init__(self, cfg: dict, img_size: int = 640) -> None:
        self.img_size = img_size
        self.p_hflip = cfg.get("horizontal_flip", 0.5)
        self.p_vflip = cfg.get("vertical_flip", 0.3)
        self.p_color = cfg.get("color_jitter", 0.8)
        self.p_mosaic = cfg.get("mosaic", 0.5)
        self.p_mixup = cfg.get("mixup", 0.1)
        self.p_scale = cfg.get("random_scale", 0.5)
        self.p_blur = cfg.get("gaussian_blur", 0.2)

    def apply(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dataset: "DroneDataset | None" = None,  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the full augmentation pipeline to one (image, boxes) pair.

        Args:
            image:   (H, W, 3) uint8 BGR image (already letterboxed).
            boxes:   (N, 5) [cls, cx, cy, w, h] normalized.
            dataset: The DroneDataset instance (needed for Mosaic/MixUp
                     which require additional random samples).

        Returns:
            (augmented_image, augmented_boxes)
        """
        # Mosaic (applied before all other transforms — changes image size)
        if dataset is not None and random.random() < self.p_mosaic:
            image, boxes = self._mosaic(image, boxes, dataset)

        # Random scale
        if random.random() < self.p_scale:
            image, boxes = self._random_scale(image, boxes)

        # Color jitter
        if random.random() < self.p_color:
            image = self._color_jitter(image)

        # Gaussian blur
        if random.random() < self.p_blur:
            image = self._gaussian_blur(image)

        # Horizontal flip
        if random.random() < self.p_hflip:
            image, boxes = self._hflip(image, boxes)

        # Vertical flip
        if random.random() < self.p_vflip:
            image, boxes = self._vflip(image, boxes)

        # MixUp (applied last — combines with another image)
        if dataset is not None and random.random() < self.p_mixup:
            image, boxes = self._mixup(image, boxes, dataset)

        return image, boxes

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _hflip(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal flip — mirror cx coordinate."""
        image = cv2.flip(image, 1)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 1] = 1.0 - boxes[:, 1]  # cx → 1 - cx
        return image, boxes

    def _vflip(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vertical flip — mirror cy coordinate."""
        image = cv2.flip(image, 0)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 2] = 1.0 - boxes[:, 2]  # cy → 1 - cy
        return image, boxes

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random HSV jitter for brightness, contrast, and saturation.

        Uses HSV color space for perceptually uniform adjustment.
        """
        # Random HSV gains
        h_gain = 1.0 + random.uniform(-0.015, 0.015)   # hue
        s_gain = 1.0 + random.uniform(-0.7, 0.7)        # saturation
        v_gain = 1.0 + random.uniform(-0.4, 0.4)        # value

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.clip(hsv[..., 0] * h_gain, 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply mild Gaussian blur to simulate motion blur / long-range imaging."""
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)

    def _random_scale(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Random scale augmentation — simulates altitude variation.

        The image is scaled by a random factor in [0.5, 1.5] and then
        letterboxed back to img_size. This effectively simulates a drone
        appearing larger or smaller in the frame.
        """
        scale = random.uniform(0.5, 1.5)
        h, w = image.shape[:2]
        new_h = max(int(h * scale), 1)
        new_w = max(int(w * scale), 1)

        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox back to img_size
        from dronevision.data.transforms import Letterbox
        lb = Letterbox(target_size=self.img_size)
        scaled, boxes, _ = lb(scaled, boxes)

        return scaled, boxes if boxes is not None else np.zeros((0, 5), dtype=np.float32)

    def _mosaic(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dataset: "DroneDataset",  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        4-image mosaic augmentation.

        Randomly samples 3 additional images from the dataset and places
        all 4 images in the quadrants of a 2×img_size canvas, then crops
        a random img_size×img_size region from the center.

        Critical for small drone detection: a drone that appears small in
        one image will appear in a compressed quadrant, further testing
        the model's ability to detect tiny targets.
        """
        # Random mosaic center within the center 50% of the canvas
        canvas_size = self.img_size * 2
        cx = int(canvas_size * random.uniform(0.25, 0.75))
        cy = int(canvas_size * random.uniform(0.25, 0.75))

        canvas = np.full((canvas_size, canvas_size, 3), 114, dtype=np.uint8)
        all_boxes: list[np.ndarray] = []

        # Get 3 random additional samples
        n = len(dataset)
        indices = [random.randint(0, n - 1) for _ in range(3)]
        images_boxes = [(image, boxes)]
        for idx in indices:
            img_i, boxes_i, _ = dataset.load_raw(idx)
            images_boxes.append((img_i, boxes_i))

        # Quadrant placements: top-left, top-right, bottom-left, bottom-right
        placements = [
            (0, 0, cx, cy),           # top-left:     x:[0,cx],    y:[0,cy]
            (cx, 0, canvas_size, cy), # top-right:    x:[cx,W],    y:[0,cy]
            (0, cy, cx, canvas_size), # bottom-left:  x:[0,cx],    y:[cy,H]
            (cx, cy, canvas_size, canvas_size),  # bottom-right
        ]

        for (img_i, boxes_i), (x1, y1, x2, y2) in zip(images_boxes, placements):
            cell_w = x2 - x1
            cell_h = y2 - y1
            img_resized = cv2.resize(img_i, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            canvas[y1:y2, x1:x2] = img_resized

            if boxes_i is not None and len(boxes_i) > 0:
                b = boxes_i.copy().astype(np.float32)
                # Map normalized box to canvas coordinates
                b[:, 1] = b[:, 1] * cell_w + x1
                b[:, 2] = b[:, 2] * cell_h + y1
                b[:, 3] = b[:, 3] * cell_w
                b[:, 4] = b[:, 4] * cell_h
                all_boxes.append(b)

        # Crop random img_size region centered on (cx, cy)
        crop_x1 = cx - self.img_size // 2
        crop_y1 = cy - self.img_size // 2
        crop_x2 = crop_x1 + self.img_size
        crop_y2 = crop_y1 + self.img_size

        # Clamp crop to canvas bounds
        crop_x1 = max(crop_x1, 0)
        crop_y1 = max(crop_y1, 0)
        crop_x2 = min(crop_x2, canvas_size)
        crop_y2 = min(crop_y2, canvas_size)

        mosaic_img = canvas[crop_y1:crop_y2, crop_x1:crop_x2]
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))

        # Adjust box coordinates to crop and re-normalize
        if all_boxes:
            all_b = np.concatenate(all_boxes, axis=0)
            # Subtract crop origin
            all_b[:, 1] -= crop_x1
            all_b[:, 2] -= crop_y1
            # Clamp to crop bounds
            cw = crop_x2 - crop_x1
            ch = crop_y2 - crop_y1
            # xyxy clipping
            x1_b = np.clip(all_b[:, 1] - all_b[:, 3] / 2, 0, cw)
            y1_b = np.clip(all_b[:, 2] - all_b[:, 4] / 2, 0, ch)
            x2_b = np.clip(all_b[:, 1] + all_b[:, 3] / 2, 0, cw)
            y2_b = np.clip(all_b[:, 2] + all_b[:, 4] / 2, 0, ch)
            all_b[:, 1] = (x1_b + x2_b) / 2
            all_b[:, 2] = (y1_b + y2_b) / 2
            all_b[:, 3] = x2_b - x1_b
            all_b[:, 4] = y2_b - y1_b
            # Normalize by crop dimensions
            all_b[:, 1] /= cw
            all_b[:, 2] /= ch
            all_b[:, 3] /= cw
            all_b[:, 4] /= ch
            mosaic_boxes = _clip_and_filter_boxes(all_b, self.img_size, self.img_size)
        else:
            mosaic_boxes = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, mosaic_boxes

    def _mixup(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        dataset: "DroneDataset",  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        MixUp augmentation — blend two images with a random Beta(8,8) ratio.

        Unlike image classification MixUp, for detection we use a hard label
        combination: simply concatenate the two sets of boxes.
        """
        idx = random.randint(0, len(dataset) - 1)
        img2, boxes2, _ = dataset.load_raw(idx)

        # Resize img2 to same size
        img2 = cv2.resize(img2, (image.shape[1], image.shape[0]))

        # Beta(8, 8) gives ratio centered around 0.5
        ratio = np.random.beta(8.0, 8.0)
        mixed = (image * ratio + img2 * (1 - ratio)).astype(np.uint8)

        if boxes2 is not None and len(boxes2) > 0:
            combined_boxes = np.concatenate([boxes, boxes2], axis=0) if len(boxes) > 0 else boxes2
        else:
            combined_boxes = boxes

        return mixed, combined_boxes
