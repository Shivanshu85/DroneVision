"""
Image transforms for DroneVision preprocessing pipeline.

All transforms operate on (image: np.ndarray, boxes: np.ndarray) pairs
where image is HWC uint8 BGR and boxes are (N, 5) [cls, cx, cy, w, h] normalized.

The core transform is Letterbox, which resizes any input image to the
target square size while preserving aspect ratio via symmetric padding.
This is critical because:
  1. Drone datasets contain images at various aspect ratios.
  2. Distortion-based resizing would change drone shapes.
  3. Letterboxing preserves spatial relationships needed for accurate detection.
"""

from __future__ import annotations

import cv2
import numpy as np


class Letterbox:
    """
    Resize image to target square size with aspect-ratio preserving padding.

    The padding color is (114, 114, 114) — a neutral gray that minimizes
    bias in network activations on padded regions.

    Args:
        target_size: Target square dimension (e.g. 640).
        stride:      Model stride — ensures output is divisible by stride.
        pad_color:   BGR fill color for padding (default: neutral gray).
    """

    def __init__(
        self,
        target_size: int = 640,
        stride: int = 32,
        pad_color: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        self.target_size = target_size
        self.stride = stride
        self.pad_color = pad_color

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """
        Apply letterbox resize to image and adjust box coordinates.

        Args:
            image:  HWC uint8 BGR image.
            boxes:  (N, 5) [cls, cx, cy, w, h] normalized, or None.

        Returns:
            (resized_image, adjusted_boxes, meta)
            meta contains: {ratio, pad_left, pad_top, orig_h, orig_w}
        """
        orig_h, orig_w = image.shape[:2]

        # Compute uniform scale factor
        ratio = min(self.target_size / orig_h, self.target_size / orig_w)
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute symmetric padding
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=self.pad_color,
        )

        meta = {
            "ratio": ratio,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }

        # Adjust box coordinates to letterboxed image
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.copy().astype(np.float32)
            # Denormalize to original pixel coords
            boxes[:, 1] = boxes[:, 1] * orig_w  # cx pixels
            boxes[:, 2] = boxes[:, 2] * orig_h  # cy pixels
            boxes[:, 3] = boxes[:, 3] * orig_w  # w pixels
            boxes[:, 4] = boxes[:, 4] * orig_h  # h pixels

            # Apply scale and padding
            boxes[:, 1] = boxes[:, 1] * ratio + pad_left
            boxes[:, 2] = boxes[:, 2] * ratio + pad_top
            boxes[:, 3] = boxes[:, 3] * ratio
            boxes[:, 4] = boxes[:, 4] * ratio

            # Re-normalize to [0,1]
            boxes[:, 1] /= self.target_size
            boxes[:, 2] /= self.target_size
            boxes[:, 3] /= self.target_size
            boxes[:, 4] /= self.target_size

            # Clamp to valid range
            boxes[:, 1:] = np.clip(boxes[:, 1:], 0.0, 1.0)

        return padded, boxes, meta


class ToTensor:
    """
    Convert HWC uint8 BGR numpy image to CHW float32 RGB tensor.

    Args:
        normalize: If True, divide by 255 to get [0.0, 1.0] range.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W, 3) uint8 BGR image.

        Returns:
            (3, H, W) float32 array in RGB order.
        """
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # HWC → CHW
        image = image.transpose(2, 0, 1)
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        return image


def rescale_boxes(
    boxes: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Reverse a letterbox transform — map predicted boxes back to original image space.

    Args:
        boxes:  (N, 4) normalized cxcywh boxes in letterboxed image space.
        meta:   Meta dict returned by Letterbox.__call__.

    Returns:
        (N, 4) normalized cxcywh boxes in original image space.
    """
    if len(boxes) == 0:
        return boxes

    ratio = meta["ratio"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]
    target_size = orig_w * ratio + pad_left * 2  # approximate (may differ by 1px)

    boxes = boxes.copy().astype(np.float32)

    # Denormalize to letterboxed pixel coords
    boxes[:, 0] = boxes[:, 0] * target_size - pad_left
    boxes[:, 1] = boxes[:, 1] * target_size - pad_top
    boxes[:, 2] = boxes[:, 2] * target_size
    boxes[:, 3] = boxes[:, 3] * target_size

    # Undo scaling
    boxes[:, 0] /= ratio
    boxes[:, 1] /= ratio
    boxes[:, 2] /= ratio
    boxes[:, 3] /= ratio

    # Re-normalize to original image
    boxes[:, 0] /= orig_w
    boxes[:, 1] /= orig_h
    boxes[:, 2] /= orig_w
    boxes[:, 3] /= orig_h

    return np.clip(boxes, 0.0, 1.0)
