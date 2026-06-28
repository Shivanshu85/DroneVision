"""
Non-Maximum Suppression for DroneVision inference.

Filters raw decoded predictions to a set of non-overlapping detections.

Input format:  (B, N_anchors, 5+nc) decoded predictions
               where columns are [cx, cy, w, h, obj_conf, cls_conf...]
               all in normalized [0,1] coordinates.

Output format: list of B arrays, each (M_i, 6) [x1, y1, x2, y2, conf, cls_id]
               or None if no detections above threshold.
"""

from __future__ import annotations

import numpy as np
import torch

from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> list[np.ndarray | None]:
    """
    Apply confidence filtering and NMS to a batch of decoded predictions.

    Args:
        predictions:    (B, N, 5+nc) decoded predictions
                        [cx, cy, w, h, obj_conf, cls_conf...].
        conf_threshold: Minimum objectness × class confidence to keep a detection.
        iou_threshold:  IoU threshold for NMS suppression.
        max_detections: Maximum number of detections to return per image.

    Returns:
        List of B elements, each:
          - np.ndarray of shape (M, 6) [x1, y1, x2, y2, conf, cls_id], or
          - None if no detections survive filtering.
        All coordinates are normalized to [0, 1].
    """
    B = predictions.shape[0]
    results: list[np.ndarray | None] = []

    for b in range(B):
        pred = predictions[b]  # (N, 5+nc)

        # Compute detection confidence: obj × cls_max
        obj_conf = pred[:, 4]
        cls_conf, cls_ids = pred[:, 5:].max(dim=1)
        det_conf = obj_conf * cls_conf

        # Confidence filter
        mask = det_conf >= conf_threshold
        if not mask.any():
            results.append(None)
            continue

        pred_filt = pred[mask]         # (M, 5+nc)
        conf_filt = det_conf[mask]     # (M,)
        cls_filt = cls_ids[mask]       # (M,)

        # Convert cxcywh → xyxy
        boxes_cxcywh = pred_filt[:, :4]
        boxes_xyxy = torch.cat([
            boxes_cxcywh[:, :2] - boxes_cxcywh[:, 2:4] / 2,
            boxes_cxcywh[:, :2] + boxes_cxcywh[:, 2:4] / 2,
        ], dim=1)
        boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)

        # NMS (applied per class; with 1 class this is straightforward)
        keep_indices = _nms(boxes_xyxy, conf_filt, iou_threshold)

        # Limit to max_detections
        if len(keep_indices) > max_detections:
            # Keep highest confidence
            top_conf = conf_filt[keep_indices]
            keep_indices = keep_indices[top_conf.argsort(descending=True)[:max_detections]]

        kept_boxes = boxes_xyxy[keep_indices].cpu().numpy()         # (K, 4)
        kept_conf = conf_filt[keep_indices].cpu().numpy()[:, None]  # (K, 1)
        kept_cls = cls_filt[keep_indices].cpu().float().numpy()[:, None]  # (K, 1)

        det_array = np.concatenate([kept_boxes, kept_conf, kept_cls], axis=1)  # (K, 6)
        results.append(det_array if len(det_array) > 0 else None)

    return results


def _nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Greedy NMS on xyxy boxes sorted by score.

    Args:
        boxes:         (N, 4) xyxy tensor.
        scores:        (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        (K,) index tensor of kept detections.
    """
    # Sort by score descending
    order = scores.argsort(descending=True)
    boxes = boxes[order]

    keep: list[int] = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(order[i].item())
        if i == len(boxes) - 1:
            break
        # Compute IoU of box i with all remaining boxes
        iou = _box_iou(boxes[i:i+1], boxes[i+1:])  # (1, N-i-1)
        suppressed[i+1:] |= iou[0] > iou_threshold

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        box1: (M, 4) xyxy.
        box2: (N, 4) xyxy.

    Returns:
        (M, N) IoU matrix.
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    ix1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    iy1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    ix2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    iy2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union
