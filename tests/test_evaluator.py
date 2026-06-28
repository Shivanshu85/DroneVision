"""Tests for DroneEvaluator metric computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dronevision.engine.evaluator import DroneEvaluator


def _make_perfect_preds(targets: torch.Tensor) -> list[np.ndarray]:
    """Create predictions that exactly match the targets (for AP=1 case)."""
    B = max(int(targets[:, 0].max().item()) + 1 if len(targets) else 1, 1)
    preds_list = []
    for b in range(B):
        mask = targets[:, 0] == b
        t_b = targets[mask]
        if len(t_b) == 0:
            preds_list.append(None)
            continue
        # Convert cxcywh → xyxy
        cxcywh = t_b[:, 2:6].numpy()
        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2
        cls = t_b[:, 1].numpy()
        conf = np.ones(len(t_b), dtype=np.float32)
        det_arr = np.stack([x1, y1, x2, y2, conf, cls], axis=1).astype(np.float32)
        preds_list.append(det_arr)
    return preds_list


class TestDroneEvaluator:
    def test_compute_on_empty_returns_zero(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        results = ev.compute()
        assert results.map50 == 0.0

    def test_perfect_predictions_high_map(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()

        # Create ground truth
        targets = torch.tensor([
            [0, 0, 0.3, 0.4, 0.05, 0.05],  # batch 0
            [0, 0, 0.7, 0.6, 0.04, 0.06],  # batch 0
            [1, 0, 0.5, 0.5, 0.08, 0.07],  # batch 1
        ])

        preds = _make_perfect_preds(targets)
        ev.update(preds, targets)
        results = ev.compute(iou_thresholds=[0.50])
        # Perfect predictions should have AP near 1.0
        assert results.map50 > 0.8

    def test_no_predictions_zero_precision_one_recall(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
        ])
        ev.update([None], targets)
        results = ev.compute(iou_thresholds=[0.50])
        # No predictions: recall = 0, precision = 0 (no TPs), AP = 0
        assert results.map50 == pytest.approx(0.0, abs=1e-4)

    def test_false_positives_reduce_precision(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
        ])
        # Many false positive predictions far from GT
        fp_boxes = np.array([[0.1, 0.1, 0.15, 0.15, 0.9, 0.0]] * 10, dtype=np.float32)
        ev.update([fp_boxes], targets)
        results = ev.compute(iou_thresholds=[0.50])
        # Low AP expected
        assert results.map50 < 0.5

    def test_reset_clears_state(self):
        ev = DroneEvaluator(num_classes=1)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
        ev.update([None], targets)
        ev.reset()
        assert ev._img_count == 0
        assert len(ev._all_preds) == 0

    def test_update_multiple_batches(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        for _ in range(5):
            targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
            preds = _make_perfect_preds(targets)
            ev.update(preds, targets)
        assert ev._img_count == 5

    def test_map50_in_valid_range(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        targets = torch.tensor([
            [0, 0, 0.3, 0.4, 0.05, 0.05],
            [0, 0, 0.7, 0.6, 0.04, 0.06],
        ])
        preds = _make_perfect_preds(targets)
        ev.update(preds, targets)
        results = ev.compute(iou_thresholds=[0.50])
        assert 0.0 <= results.map50 <= 1.0

    def test_map50_95_computed(self):
        ev = DroneEvaluator(num_classes=1)
        ev.reset()
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
        ])
        preds = _make_perfect_preds(targets)
        ev.update(preds, targets)
        results = ev.compute()  # default: 0.50 to 0.95
        # map50_95 should be defined and <= map50
        assert 0.0 <= results.map50_95 <= results.map50 + 0.01
