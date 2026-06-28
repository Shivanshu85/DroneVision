"""
Bounding box format conversion and IoU utilities for DroneVision.

Box formats used throughout the project:
    - YOLO:  [class_id, cx, cy, w, h]      normalized [0,1]
    - cxcywh: [cx, cy, w, h]               can be normalized or pixel
    - xyxy:  [x1, y1, x2, y2]              can be normalized or pixel

All functions operate on the last dimension of input tensors, supporting
arbitrary batch dimensions (e.g., (N, 4), (B, N, 4), etc.).
"""

from __future__ import annotations

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (cx, cy, w, h) → (x1, y1, x2, y2).

    Args:
        boxes: (..., 4) tensor in center-format.

    Returns:
        (..., 4) tensor in corner-format.
    """
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return out


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (x1, y1, x2, y2) → (cx, cy, w, h).

    Args:
        boxes: (..., 4) tensor in corner-format.

    Returns:
        (..., 4) tensor in center-format.
    """
    out = boxes.clone()
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # cx
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # cy
    out[..., 2] = boxes[..., 2] - boxes[..., 0]         # w
    out[..., 3] = boxes[..., 3] - boxes[..., 1]         # h
    return out


def yolo_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Convert normalized YOLO (cx, cy, w, h) → pixel (x1, y1, x2, y2).

    Args:
        boxes:  (..., 4) normalized cxcywh tensor.
        img_w:  Image width in pixels.
        img_h:  Image height in pixels.

    Returns:
        (..., 4) pixel xyxy tensor.
    """
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return cxcywh_to_xyxy(boxes) * scale


def xyxy_to_yolo(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Convert pixel (x1, y1, x2, y2) → normalized YOLO (cx, cy, w, h).

    Args:
        boxes:  (..., 4) pixel xyxy tensor.
        img_w:  Image width in pixels.
        img_h:  Image height in pixels.

    Returns:
        (..., 4) normalized cxcywh tensor.
    """
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return xyxy_to_cxcywh(boxes / scale)


# ---------------------------------------------------------------------------
# IoU family
# ---------------------------------------------------------------------------

def compute_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise IoU between two sets of xyxy boxes.

    Args:
        box1: (N, 4) xyxy tensor.
        box2: (N, 4) xyxy tensor.

    Returns:
        (N,) IoU scores.
    """
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU matrix between two sets of xyxy boxes.

    Args:
        boxes1: (M, 4) xyxy tensor.
        boxes2: (N, 4) xyxy tensor.

    Returns:
        (M, N) IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (M,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (N,)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # (M,N)
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h  # (M,N)

    union_area = area1[:, None] + area2[None, :] - inter_area + 1e-7
    return inter_area / union_area


def clip_boxes_to_image(
    boxes: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """
    Clip xyxy boxes to image boundaries.

    Args:
        boxes:  (..., 4) xyxy tensor.
        img_h:  Image height.
        img_w:  Image width.

    Returns:
        Clipped (..., 4) xyxy tensor.
    """
    out = boxes.clone()
    out[..., 0].clamp_(0, img_w)
    out[..., 1].clamp_(0, img_h)
    out[..., 2].clamp_(0, img_w)
    out[..., 3].clamp_(0, img_h)
    return out


def filter_small_boxes(
    boxes: torch.Tensor,
    min_size: float = 2.0,
) -> torch.Tensor:
    """
    Return a boolean mask for boxes whose width and height are >= min_size.

    Args:
        boxes:    (N, 4) xyxy tensor.
        min_size: Minimum side length in pixels.

    Returns:
        (N,) boolean tensor — True = keep.
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w >= min_size) & (h >= min_size)
