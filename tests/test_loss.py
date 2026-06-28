"""Tests for IoU loss functions and DroneDetectionLoss."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from dronevision.loss.iou_loss import compute_iou, compute_ciou, ciou_loss
from dronevision.loss.detection_loss import DroneDetectionLoss
from dronevision.models.detector import DroneDetector


class TestIoULoss:
    def test_perfect_iou_is_one(self):
        box = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        iou = compute_iou(box, box)
        assert iou.item() == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap_iou_is_zero(self):
        box1 = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        box2 = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        iou = compute_iou(box1, box2)
        assert iou.item() == pytest.approx(0.0, abs=1e-5)

    def test_iou_in_unit_range(self):
        torch.manual_seed(0)
        a = torch.rand(100, 4)
        a[:, 2:] = a[:, 2:] * 0.3 + 0.01  # ensure positive wh
        b = torch.rand(100, 4)
        b[:, 2:] = b[:, 2:] * 0.3 + 0.01
        iou = compute_iou(a, b)
        assert (iou >= 0).all()
        assert (iou <= 1 + 1e-5).all()

    def test_perfect_ciou_is_one(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        ciou = compute_ciou(box, box)
        assert ciou.item() == pytest.approx(1.0, abs=1e-4)

    def test_ciou_loss_zero_for_perfect(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        loss = ciou_loss(box, box)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_ciou_loss_positive(self):
        box1 = torch.tensor([[0.3, 0.3, 0.1, 0.1]])
        box2 = torch.tensor([[0.7, 0.7, 0.2, 0.2]])
        loss = ciou_loss(box1, box2)
        assert loss.item() > 0.0

    def test_ciou_loss_batch(self):
        torch.manual_seed(42)
        pred = torch.rand(32, 4)
        pred[:, 2:] = pred[:, 2:] * 0.2 + 0.02
        target = torch.rand(32, 4)
        target[:, 2:] = target[:, 2:] * 0.2 + 0.02
        loss = ciou_loss(pred, target)
        assert loss.shape == (32,)
        assert not torch.isnan(loss).any()


class TestDroneDetectionLoss:
    def _make_raw_preds(self, cfg: dict, batch_size: int = 2) -> list[torch.Tensor]:
        """Create random raw predictions matching head output shapes for 416×416."""
        img = cfg["model"]["image_size"]
        na = cfg["model"]["num_anchors"]
        nc = cfg["model"]["num_classes"]
        strides = cfg["model"]["strides"]
        preds = []
        for stride in strides:
            H = img // stride
            W = img // stride
            preds.append(torch.randn(batch_size, na, H, W, 5 + nc))
        return preds

    def test_loss_returns_finite(self, dev_cfg, batch_targets):
        criterion = DroneDetectionLoss(dev_cfg)
        model = DroneDetector(dev_cfg)
        preds = self._make_raw_preds(dev_cfg)
        total_loss, metrics = criterion(preds, batch_targets, model.anchors)
        assert torch.isfinite(total_loss)

    def test_loss_components_in_metrics(self, dev_cfg, batch_targets):
        criterion = DroneDetectionLoss(dev_cfg)
        model = DroneDetector(dev_cfg)
        preds = self._make_raw_preds(dev_cfg)
        _, metrics = criterion(preds, batch_targets, model.anchors)
        assert "box_loss" in metrics
        assert "obj_loss" in metrics
        assert "cls_loss" in metrics
        assert "total_loss" in metrics

    def test_loss_positive(self, dev_cfg, batch_targets):
        criterion = DroneDetectionLoss(dev_cfg)
        model = DroneDetector(dev_cfg)
        preds = self._make_raw_preds(dev_cfg)
        total_loss, _ = criterion(preds, batch_targets, model.anchors)
        assert total_loss.item() > 0.0

    def test_loss_with_empty_targets(self, dev_cfg):
        """Empty batch (all background) should not error."""
        criterion = DroneDetectionLoss(dev_cfg)
        model = DroneDetector(dev_cfg)
        empty_targets = torch.zeros(0, 6)
        preds = self._make_raw_preds(dev_cfg)
        total_loss, metrics = criterion(preds, empty_targets, model.anchors)
        assert torch.isfinite(total_loss)
        assert total_loss.item() >= 0.0

    def test_gradient_flows_through_loss(self, dev_cfg, batch_targets):
        criterion = DroneDetectionLoss(dev_cfg)
        model = DroneDetector(dev_cfg)
        model.train()
        x = torch.randn(2, 3, 416, 416)
        preds = model(x)
        total_loss, _ = criterion(preds, batch_targets, model.anchors)
        total_loss.backward()
        # Verify at least one parameter has a non-None gradient
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads
