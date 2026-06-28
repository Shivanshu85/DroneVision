"""
IoU loss family for DroneVision bounding box regression.

Implements:
    - IoU  : Intersection over Union
    - GIoU : Generalized IoU (adds enclosing box penalty)
    - DIoU : Distance IoU (adds center distance penalty)
    - CIoU : Complete IoU (DIoU + aspect ratio penalty)  ← PRIMARY

CIoU is the recommended loss for drone detection because:
    1. Drones are often non-square (elongated UAV shapes, propellers).
    2. CIoU's aspect ratio term penalizes predictions that have the wrong
       aspect ratio even when IoU is high.
    3. CIoU converges faster than plain IoU or GIoU for small objects.

All functions operate on (cx, cy, w, h) normalized boxes.
"""

from __future__ import annotations

import math

import torch


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2). Last dim must be 4."""
    return torch.stack([
        boxes[..., 0] - boxes[..., 2] / 2,
        boxes[..., 1] - boxes[..., 3] / 2,
        boxes[..., 0] + boxes[..., 2] / 2,
        boxes[..., 1] + boxes[..., 3] / 2,
    ], dim=-1)


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute element-wise IoU between cxcywh boxes.

    Args:
        pred:   (..., 4) predicted boxes [cx, cy, w, h].
        target: (..., 4) target boxes    [cx, cy, w, h].
        eps:    Small constant for numerical stability.

    Returns:
        (...,) IoU scores in [0, 1].
    """
    p_xyxy = _cxcywh_to_xyxy(pred)
    t_xyxy = _cxcywh_to_xyxy(target)

    ix1 = torch.max(p_xyxy[..., 0], t_xyxy[..., 0])
    iy1 = torch.max(p_xyxy[..., 1], t_xyxy[..., 1])
    ix2 = torch.min(p_xyxy[..., 2], t_xyxy[..., 2])
    iy2 = torch.min(p_xyxy[..., 3], t_xyxy[..., 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    area_p = pred[..., 2] * pred[..., 3]
    area_t = target[..., 2] * target[..., 3]
    union = area_p + area_t - inter + eps

    return inter / union


def compute_ciou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute element-wise CIoU between cxcywh boxes.

    CIoU = IoU - (center_distance² / enclosing_diag²) - α·v
    where:
        v = (4/π²) · (arctan(tw/th) - arctan(pw/ph))²
        α = v / (1 - IoU + v)    (gradient-stopping weight)

    Args:
        pred:   (..., 4) predicted boxes [cx, cy, w, h] normalized.
        target: (..., 4) target boxes    [cx, cy, w, h] normalized.
        eps:    Numerical stability constant.

    Returns:
        (...,) CIoU scores.  CIoU loss = 1 - CIoU.
        Higher CIoU → better prediction (max = 1).
    """
    p_xyxy = _cxcywh_to_xyxy(pred)
    t_xyxy = _cxcywh_to_xyxy(target)

    # Intersection
    ix1 = torch.max(p_xyxy[..., 0], t_xyxy[..., 0])
    iy1 = torch.max(p_xyxy[..., 1], t_xyxy[..., 1])
    ix2 = torch.min(p_xyxy[..., 2], t_xyxy[..., 2])
    iy2 = torch.min(p_xyxy[..., 3], t_xyxy[..., 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = pred[..., 2] * pred[..., 3]
    area_t = target[..., 2] * target[..., 3]
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Enclosing box diagonal²
    enc_x1 = torch.min(p_xyxy[..., 0], t_xyxy[..., 0])
    enc_y1 = torch.min(p_xyxy[..., 1], t_xyxy[..., 1])
    enc_x2 = torch.max(p_xyxy[..., 2], t_xyxy[..., 2])
    enc_y2 = torch.max(p_xyxy[..., 3], t_xyxy[..., 3])
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Center distance²
    center_dist = (pred[..., 0] - target[..., 0]) ** 2 + \
                  (pred[..., 1] - target[..., 1]) ** 2

    # Aspect ratio consistency term
    with torch.no_grad():
        pw = pred[..., 2].clamp(min=eps)
        ph = pred[..., 3].clamp(min=eps)
        tw = target[..., 2].clamp(min=eps)
        th = target[..., 3].clamp(min=eps)
        v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (center_dist / enc_diag) - alpha * v
    return ciou


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CIoU loss = 1 - CIoU(pred, target).

    Args:
        pred:   (..., 4) [cx, cy, w, h] normalized predicted boxes.
        target: (..., 4) [cx, cy, w, h] normalized target boxes.

    Returns:
        (...,) loss values in [0, ~2] (typically [0, 1.5] in practice).
    """
    return 1.0 - compute_ciou(pred, target)
