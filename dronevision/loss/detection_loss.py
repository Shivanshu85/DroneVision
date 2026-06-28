"""
DroneDetectionLoss — Combined detection loss for DroneVision.

Loss components:
    L_box : CIoU loss on bounding box regression (positive anchors only).
    L_obj : BCE loss on objectness confidence (all anchors).
    L_cls : BCE loss on class confidence (positive anchors only, always class 0).

Total loss:
    L = λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls

Target assignment:
    For each GT box, we assign it to any anchor whose aspect ratio is within
    a threshold (default: 4×) of the GT box shape. Multiple anchors per scale
    can be assigned to the same GT box (if multiple anchors pass the ratio test),
    which improves recall at the cost of slightly more positive samples.

    This is analogous to YOLO's multi-anchor assignment but simplified:
    - No neighboring cell assignment (for Phase 1 simplicity).
    - No soft targets for class confidence (always 1 for drone).
    - Objectness target uses IoU score as soft label (improves NMS quality).

Object/background balance:
    With thousands of anchor locations per image and typically very few drones,
    objectness loss must be weighted carefully.
    Scale-dependent balance weights [4.0, 1.0, 0.4] for [P3, P4, P5] are used:
    - P3 (small scale) has 80×80 = 6400 cells → most background → highest weight
    - P5 (large scale) has 20×20 = 400 cells  → fewer background → lower weight
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dronevision.loss.iou_loss import ciou_loss, compute_ciou
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)

# Scale objectness balance weights [P3, P4, P5]
# Upweights small-scale (P3) to compensate for its larger grid size
_OBJ_BALANCE: list[float] = [4.0, 1.0, 0.4]


class DroneDetectionLoss(nn.Module):
    """
    Combined detection loss for single-class drone detection.

    Args:
        cfg: Full configuration dict. Uses cfg["loss"] and cfg["model"] sections.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        loss_cfg = cfg["loss"]
        model_cfg = cfg["model"]

        self.num_classes: int = model_cfg["num_classes"]
        self.num_anchors: int = model_cfg["num_anchors"]
        self.img_size: int = model_cfg["image_size"]
        self.strides: list[int] = model_cfg["strides"]
        self.anchor_threshold: float = loss_cfg["anchor_threshold"]

        self.lambda_box: float = loss_cfg["lambda_box"]
        self.lambda_obj: float = loss_cfg["lambda_obj"]
        self.lambda_cls: float = loss_cfg["lambda_cls"]

        # Objectness BCE with positive weighting
        obj_pw = loss_cfg.get("obj_pos_weight", 1.0)
        cls_pw = loss_cfg.get("cls_pos_weight", 1.0)
        self.bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([obj_pw])
        )
        self.bce_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cls_pw])
        )

        logger.info(
            "DroneDetectionLoss | λ_box=%.1f | λ_obj=%.1f | λ_cls=%.1f | "
            "anchor_thresh=%.1f",
            self.lambda_box, self.lambda_obj, self.lambda_cls,
            self.anchor_threshold,
        )

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the combined detection loss.

        Args:
            predictions: List of 3 raw prediction tensors from DroneHead, each
                         (B, num_anchors, H_i, W_i, 5+nc).
            targets:     (N, 6) float tensor [batch_idx, cls, cx, cy, w, h]
                         normalized to [0,1]. Empty (0,6) for all-background batches.
            anchors:     (3, 3, 2) anchor [w, h] tensor in pixel space (from model buffer).

        Returns:
            (total_loss, metrics_dict)
            metrics_dict contains: box_loss, obj_loss, cls_loss, total_loss.
        """
        device = predictions[0].device

        # Move BCE loss pos_weights to device
        self.bce_obj.pos_weight = self.bce_obj.pos_weight.to(device)
        self.bce_cls.pos_weight = self.bce_cls.pos_weight.to(device)

        # Build assignment targets
        indices, tboxes, anchors_matched, tclasses = self._build_targets(
            predictions, targets, anchors
        )

        total_box_loss = torch.zeros(1, device=device)
        total_obj_loss = torch.zeros(1, device=device)
        total_cls_loss = torch.zeros(1, device=device)

        for scale_idx, pred in enumerate(predictions):
            B, na, H, W, nc5 = pred.shape
            stride = self.strides[scale_idx]

            b_idx, a_idx, gj, gi = indices[scale_idx]
            n_pos = len(b_idx)

            # Objectness target: zeros for all anchors initially
            target_obj = torch.zeros(B, na, H, W, 1, device=device)

            if n_pos > 0:
                # Select positive anchor predictions
                pred_pos = pred[b_idx, a_idx, gj, gi]  # (n_pos, 5+nc)

                # ── Box loss ──────────────────────────────────────────────
                pred_box = self._decode_box(
                    pred_pos[:, :4], a_idx, anchors[scale_idx], gi, gj, H, W, stride
                )  # (n_pos, 4) normalized cxcywh

                target_box = tboxes[scale_idx]  # (n_pos, 4) normalized cxcywh

                box_loss = ciou_loss(pred_box, target_box).mean()
                total_box_loss += box_loss

                # ── Soft objectness target (CIoU score) ───────────────────
                with torch.no_grad():
                    ciou_score = compute_ciou(
                        pred_box.detach(), target_box
                    ).clamp(0.0, 1.0)
                target_obj[b_idx, a_idx, gj, gi, 0] = ciou_score.float()

                # ── Class loss ────────────────────────────────────────────
                if self.num_classes > 1:
                    t_cls = torch.zeros(
                        n_pos, self.num_classes, device=device
                    )
                    t_cls[range(n_pos), tclasses[scale_idx]] = 1.0
                    cls_loss = self.bce_cls(pred_pos[:, 5:], t_cls)
                else:
                    # Single class: cls target is always 1.0 (is a drone)
                    t_cls = torch.ones(n_pos, 1, device=device)
                    cls_loss = self.bce_cls(pred_pos[:, 5:6], t_cls)
                total_cls_loss += cls_loss

            # ── Objectness loss (all anchors) ─────────────────────────────
            obj_loss = self.bce_obj(
                pred[..., 4:5], target_obj
            ) * _OBJ_BALANCE[scale_idx]
            total_obj_loss += obj_loss

        total_loss = (
            self.lambda_box * total_box_loss
            + self.lambda_obj * total_obj_loss
            + self.lambda_cls * total_cls_loss
        )

        metrics = {
            "box_loss": total_box_loss.item(),
            "obj_loss": total_obj_loss.item(),
            "cls_loss": total_cls_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def _decode_box(
        self,
        raw_box: torch.Tensor,
        a_idx: torch.Tensor,
        anchor_set: torch.Tensor,
        gi: torch.Tensor,
        gj: torch.Tensor,
        H: int,
        W: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Decode raw box predictions to normalized cxcywh.

        YOLOv5-style decoding:
            bx = (sigmoid(tx)*2 - 0.5 + gi) / W
            by = (sigmoid(ty)*2 - 0.5 + gj) / H
            bw = (sigmoid(tw)*2)^2 * anchor_w / img_size
            bh = (sigmoid(th)*2)^2 * anchor_h / img_size

        Args:
            raw_box:    (n_pos, 4) raw tx,ty,tw,th.
            a_idx:      (n_pos,) anchor indices.
            anchor_set: (3, 2) anchor [w,h] pixels for this scale.
            gi:         (n_pos,) grid x indices.
            gj:         (n_pos,) grid y indices.
            H, W:       Grid height and width.
            stride:     This scale's stride.

        Returns:
            (n_pos, 4) normalized cxcywh in [0,1].
        """
        device = raw_box.device
        anch = anchor_set[a_idx].to(device)  # (n_pos, 2)

        sig_xy = torch.sigmoid(raw_box[:, :2]) * 2 - 0.5
        bx = (sig_xy[:, 0] + gi.float()) / W
        by = (sig_xy[:, 1] + gj.float()) / H

        sig_wh = (torch.sigmoid(raw_box[:, 2:4]) * 2) ** 2
        bw = sig_wh[:, 0] * anch[:, 0] / self.img_size
        bh = sig_wh[:, 1] * anch[:, 1] / self.img_size

        return torch.stack([bx, by, bw, bh], dim=1)

    def _build_targets(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[list, list, list, list]:
        """
        Assign ground-truth boxes to anchor slots across all 3 scales.

        Args:
            predictions: List of 3 raw prediction tensors.
            targets:     (N, 6) [batch_idx, cls, cx, cy, w, h] normalized.
            anchors:     (3, 3, 2) pixel-space anchor tensor.

        Returns:
            4 lists (one entry per scale):
              - indices:  (b_tensor, a_tensor, gj_tensor, gi_tensor)
              - tboxes:   (n_pos, 4) normalized cxcywh target boxes
              - anch_matched: not used (for future extension)
              - tclasses: (n_pos,) int64 class indices
        """
        device = targets.device
        n_targets = len(targets)

        all_indices: list[tuple] = []
        all_tboxes: list[torch.Tensor] = []
        all_anch: list[torch.Tensor] = []
        all_tcls: list[torch.Tensor] = []

        _EMPTY = (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

        for scale_idx, pred in enumerate(predictions):
            B, na, H, W, _ = pred.shape
            anchor_set = anchors[scale_idx].to(device)  # (3, 2) pixels

            if n_targets == 0:
                all_indices.append(_EMPTY)
                all_tboxes.append(torch.zeros(0, 4, device=device))
                all_anch.append(torch.zeros(0, 2, device=device))
                all_tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            # Normalize anchors by image size for ratio comparison
            anch_norm = anchor_set / self.img_size  # (3, 2)

            # targets[:, 4:6] = (w_norm, h_norm)
            gt_wh = targets[:, 4:6]  # (N, 2)

            # Compute aspect ratio between each GT and each anchor
            # Expand for broadcasting: (na, N, 2)
            ratio = gt_wh[None] / (anch_norm[:, None] + 1e-8)  # (3, N, 2)
            max_ratio = torch.max(ratio, 1.0 / ratio).max(dim=2).values  # (3, N)

            # Boolean mask: which (anchor, GT) pairs pass the threshold
            mask = max_ratio < self.anchor_threshold  # (3, N)

            if not mask.any():
                all_indices.append(_EMPTY)
                all_tboxes.append(torch.zeros(0, 4, device=device))
                all_anch.append(torch.zeros(0, 2, device=device))
                all_tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            # Expand targets for each anchor: (3, N, 6) → filter
            t_exp = targets[None].expand(na, -1, -1)  # (3, N, 6)
            t_filtered = t_exp[mask]                   # (K, 6)

            # Anchor indices for each positive
            a_indices = torch.arange(na, device=device).view(na, 1)
            a_indices = a_indices.expand(-1, n_targets)  # (3, N)
            a_filtered = a_indices[mask]                 # (K,)

            # Extract components
            batch_ids = t_filtered[:, 0].long()
            cls_ids = t_filtered[:, 1].long()
            gxy_norm = t_filtered[:, 2:4]               # (K, 2) cx, cy normalized
            gwh_norm = t_filtered[:, 4:6]               # (K, 2) w, h normalized

            # Convert to grid coordinates
            gxy_grid = gxy_norm * torch.tensor([W, H], dtype=torch.float, device=device)
            gij = gxy_grid.long()
            gi_ = gij[:, 0].clamp(0, W - 1)
            gj_ = gij[:, 1].clamp(0, H - 1)

            # Target box: [offset_cx_in_cell, offset_cy_in_cell, w_norm, h_norm]
            # The loss will use these directly for CIoU computation
            # We reconstruct normalized cxcywh from the grid assignment
            tcx = (gij[:, 0].float() + 0.5) / W   # cell center cx (proxy for target)
            tcy = (gij[:, 1].float() + 0.5) / H   # cell center cy (proxy)
            # Actually store the exact GT cxcywh for accurate CIoU:
            tbox = torch.cat([gxy_norm, gwh_norm], dim=1)  # (K, 4) exact normalized cxcywh

            all_indices.append((batch_ids, a_filtered, gj_, gi_))
            all_tboxes.append(tbox)
            all_anch.append(anchor_set[a_filtered])
            all_tcls.append(cls_ids)

        return all_indices, all_tboxes, all_anch, all_tcls
