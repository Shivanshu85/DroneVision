"""
DroneEvaluator — Computes mAP50, mAP50-95, Precision, Recall, F1.

Implements the Pascal VOC / COCO evaluation protocol for single-class detection:
    1. Collect all predictions across the validation set with confidence scores.
    2. Sort predictions by confidence (descending).
    3. For each prediction, check if it matches a GT box (IoU > threshold,
       GT not already matched → TP; else FP).
    4. Compute precision-recall curve.
    5. Compute AP = area under the PR curve (11-point interpolation for mAP50,
       continuous integration for mAP50-95).

For mAP50-95: average AP over IoU thresholds {0.50, 0.55, ..., 0.95}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dronevision.utils.bbox import compute_iou_matrix
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics at one IoU threshold."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap: float = 0.0
    num_gt: int = 0
    num_pred: int = 0


@dataclass
class EvalResults:
    """Full evaluation result set."""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class: dict[int, EvalMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mAP50": self.map50,
            "mAP50-95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __str__(self) -> str:
        return (
            f"mAP50={self.map50:.4f} | mAP50-95={self.map50_95:.4f} | "
            f"P={self.precision:.4f} | R={self.recall:.4f} | F1={self.f1:.4f}"
        )


class DroneEvaluator:
    """
    Computes detection metrics over a validation set.

    Usage:
        evaluator = DroneEvaluator(num_classes=1)
        evaluator.reset()
        for images, targets, paths in val_loader:
            preds = model(images)
            decoded = nms(preds, conf_thresh, iou_thresh)
            evaluator.update(decoded, targets)
        results = evaluator.compute()
    """

    def __init__(self, num_classes: int = 1) -> None:
        self.num_classes = num_classes
        self._all_preds: list[np.ndarray] = []   # list of (N, 6) [x1,y1,x2,y2,conf,cls]
        self._all_targets: list[np.ndarray] = [] # list of (M, 5) [cls,x1,y1,x2,y2]
        self._img_count: int = 0

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self._all_preds = []
        self._all_targets = []
        self._img_count = 0

    def update(
        self,
        preds: list[np.ndarray | None],
        targets: torch.Tensor,
    ) -> None:
        """
        Accumulate one batch of predictions and ground truths.

        Args:
            preds:   List of B arrays, each (N_i, 6) [x1,y1,x2,y2,conf,cls]
                     in normalized [0,1] coords, or None for images with no dets.
            targets: (N_total, 6) [batch_idx, cls, cx, cy, w, h] normalized,
                     as returned by drone_collate_fn.
        """
        B = len(preds)

        for b in range(B):
            # Predictions for this image
            pred_b = preds[b]
            if pred_b is None or len(pred_b) == 0:
                self._all_preds.append(np.zeros((0, 6), dtype=np.float32))
            else:
                self._all_preds.append(np.asarray(pred_b, dtype=np.float32))

            # Targets for this image
            mask = targets[:, 0] == b
            t_b = targets[mask]  # (M, 6)
            if len(t_b) == 0:
                self._all_targets.append(np.zeros((0, 5), dtype=np.float32))
            else:
                # Convert cxcywh → xyxy, drop batch_idx
                cls_col = t_b[:, 1:2]
                cxcywh = t_b[:, 2:6]
                xyxy = torch.cat([
                    cxcywh[:, :2] - cxcywh[:, 2:4] / 2,
                    cxcywh[:, :2] + cxcywh[:, 2:4] / 2,
                ], dim=1)
                t_arr = torch.cat([cls_col, xyxy], dim=1).cpu().numpy()
                self._all_targets.append(t_arr.astype(np.float32))

            self._img_count += 1

    def compute(
        self,
        iou_thresholds: list[float] | None = None,
    ) -> EvalResults:
        """
        Compute mAP50, mAP50-95, Precision, Recall, F1.

        Args:
            iou_thresholds: IoU thresholds for mAP computation.
                            Default: [0.50, 0.55, ..., 0.95] (COCO style).

        Returns:
            EvalResults with all metrics.
        """
        if iou_thresholds is None:
            iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

        if self._img_count == 0:
            logger.warning("DroneEvaluator.compute() called with 0 images")
            return EvalResults()

        aps_per_threshold: list[float] = []
        ap50: float = 0.0
        prec50: float = 0.0
        rec50: float = 0.0

        for thresh_idx, iou_thresh in enumerate(iou_thresholds):
            metrics = self._compute_ap_at_threshold(iou_thresh)
            aps_per_threshold.append(metrics.ap)
            if abs(iou_thresh - 0.50) < 1e-6:
                ap50 = metrics.ap
                prec50 = metrics.precision
                rec50 = metrics.recall

        map50_95 = float(np.mean(aps_per_threshold)) if aps_per_threshold else 0.0
        f1 = (
            2 * prec50 * rec50 / (prec50 + rec50 + 1e-7)
            if (prec50 + rec50) > 0
            else 0.0
        )

        results = EvalResults(
            map50=ap50,
            map50_95=map50_95,
            precision=prec50,
            recall=rec50,
            f1=f1,
        )
        logger.info("Evaluation results: %s", results)
        return results

    def _compute_ap_at_threshold(self, iou_thresh: float) -> EvalMetrics:
        """
        Compute AP, Precision, Recall at a single IoU threshold.

        Uses the VOC 2010+ (continuous) interpolation for AP.
        """
        # Collect all predictions with image index
        all_det_conf: list[float] = []
        all_det_tp: list[int] = []
        all_det_fp: list[int] = []
        total_gt: int = 0

        for img_idx, (pred_arr, tgt_arr) in enumerate(
            zip(self._all_preds, self._all_targets)
        ):
            n_gt = len(tgt_arr)
            total_gt += n_gt
            matched_gt = np.zeros(n_gt, dtype=bool)

            if len(pred_arr) == 0:
                continue

            # Sort predictions by confidence descending
            sort_order = np.argsort(-pred_arr[:, 4])
            preds_sorted = pred_arr[sort_order]

            for det in preds_sorted:
                conf = float(det[4])
                det_box = det[:4]  # (4,) xyxy

                if n_gt == 0:
                    all_det_conf.append(conf)
                    all_det_tp.append(0)
                    all_det_fp.append(1)
                    continue

                # Compute IoU with all unmatched GT boxes
                det_t = torch.from_numpy(det_box[None]).float()
                gt_t = torch.from_numpy(tgt_arr[:, 1:]).float()  # xyxy
                iou_mat = compute_iou_matrix(det_t, gt_t)  # (1, n_gt)
                iou_vals = iou_mat[0].numpy()

                best_gt_idx = int(np.argmax(iou_vals))
                best_iou = float(iou_vals[best_gt_idx])

                all_det_conf.append(conf)
                if best_iou >= iou_thresh and not matched_gt[best_gt_idx]:
                    matched_gt[best_gt_idx] = True
                    all_det_tp.append(1)
                    all_det_fp.append(0)
                else:
                    all_det_tp.append(0)
                    all_det_fp.append(1)

        if total_gt == 0:
            return EvalMetrics()

        if not all_det_conf:
            return EvalMetrics(num_gt=total_gt, num_pred=0)

        # Sort all detections by confidence
        sort_order = np.argsort(-np.array(all_det_conf))
        tp_arr = np.array(all_det_tp)[sort_order]
        fp_arr = np.array(all_det_fp)[sort_order]

        tp_cumsum = np.cumsum(tp_arr)
        fp_cumsum = np.cumsum(fp_arr)

        recalls = tp_cumsum / (total_gt + 1e-7)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        # AP: area under PR curve (continuous interpolation)
        ap = self._compute_ap_continuous(recalls, precisions)

        # Precision/Recall at threshold (last value on curve)
        final_prec = float(precisions[-1]) if len(precisions) else 0.0
        final_rec = float(recalls[-1]) if len(recalls) else 0.0

        return EvalMetrics(
            precision=final_prec,
            recall=final_rec,
            f1=2 * final_prec * final_rec / (final_prec + final_rec + 1e-7),
            ap=ap,
            num_gt=total_gt,
            num_pred=len(all_det_conf),
        )

    @staticmethod
    def _compute_ap_continuous(
        recalls: np.ndarray,
        precisions: np.ndarray,
    ) -> float:
        """
        Compute AP using continuous area under the PR curve.

        Prepend (0, 1) and append (1, 0) to ensure a complete curve.
        """
        mrec = np.concatenate([[0.0], recalls, [1.0]])
        mprec = np.concatenate([[1.0], precisions, [0.0]])

        # Make precision monotonically decreasing from right
        for i in range(len(mprec) - 2, -1, -1):
            mprec[i] = max(mprec[i], mprec[i + 1])

        # Find points where recall changes
        change_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mprec[change_idx + 1]))
        return ap
