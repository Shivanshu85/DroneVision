"""Tests for Non-Maximum Suppression."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dronevision.inference.nms import non_max_suppression


def _make_decoded_preds(B: int = 1, N: int = 100, nc: int = 1) -> torch.Tensor:
    """Create synthetic decoded predictions (B, N, 5+nc) in [0,1]."""
    torch.manual_seed(0)
    preds = torch.rand(B, N, 5 + nc)
    # cx, cy in [0.1, 0.9], wh in [0.02, 0.15]
    preds[..., 0] = preds[..., 0] * 0.8 + 0.1
    preds[..., 1] = preds[..., 1] * 0.8 + 0.1
    preds[..., 2] = preds[..., 2] * 0.13 + 0.02
    preds[..., 3] = preds[..., 3] * 0.13 + 0.02
    # obj and cls to realistic range
    preds[..., 4] = torch.rand(B, N) * 0.5 + 0.1
    preds[..., 5] = torch.rand(B, N) * 0.5 + 0.1
    return preds


class TestNMS:
    def test_returns_list_of_length_b(self):
        preds = _make_decoded_preds(B=3)
        results = non_max_suppression(preds, conf_threshold=0.01, iou_threshold=0.45)
        assert len(results) == 3

    def test_output_is_array_or_none(self):
        preds = _make_decoded_preds(B=2)
        results = non_max_suppression(preds, conf_threshold=0.01, iou_threshold=0.45)
        for r in results:
            assert r is None or isinstance(r, np.ndarray)

    def test_high_conf_threshold_returns_none_or_fewer(self):
        preds = _make_decoded_preds(B=1, N=50)
        # Set all confidences very low
        preds[..., 4] = 0.01
        preds[..., 5] = 0.01
        results = non_max_suppression(preds, conf_threshold=0.99, iou_threshold=0.45)
        assert results[0] is None

    def test_output_format_6_columns(self):
        preds = _make_decoded_preds(B=1, N=50)
        preds[..., 4] = 0.9
        preds[..., 5] = 0.9
        results = non_max_suppression(preds, conf_threshold=0.01, iou_threshold=0.45)
        if results[0] is not None:
            assert results[0].shape[1] == 6

    def test_boxes_in_unit_range(self):
        preds = _make_decoded_preds(B=1, N=100)
        preds[..., 4] = 0.9
        preds[..., 5] = 0.9
        results = non_max_suppression(preds, conf_threshold=0.01, iou_threshold=0.45)
        if results[0] is not None:
            boxes = results[0][:, :4]
            assert boxes.min() >= 0.0 - 1e-5
            assert boxes.max() <= 1.0 + 1e-5

    def test_perfectly_overlapping_boxes_suppressed(self):
        """Two identical boxes should be suppressed to one."""
        # Two identical high-confidence boxes
        box = torch.tensor([[[0.5, 0.5, 0.1, 0.1, 0.99, 0.99],
                              [0.5, 0.5, 0.1, 0.1, 0.98, 0.98]]])
        results = non_max_suppression(box, conf_threshold=0.5, iou_threshold=0.45)
        assert results[0] is not None
        assert len(results[0]) == 1

    def test_non_overlapping_boxes_kept(self):
        """Boxes with no overlap should all be kept."""
        boxes = torch.zeros(1, 4, 6)
        # Place 4 non-overlapping boxes
        centers = [0.1, 0.3, 0.7, 0.9]
        for i, c in enumerate(centers):
            boxes[0, i] = torch.tensor([c, c, 0.05, 0.05, 0.99, 0.99])
        results = non_max_suppression(boxes, conf_threshold=0.5, iou_threshold=0.45)
        assert results[0] is not None
        assert len(results[0]) == 4

    def test_batch_processing(self):
        preds = _make_decoded_preds(B=4)
        preds[..., 4] = 0.9
        preds[..., 5] = 0.9
        results = non_max_suppression(preds, conf_threshold=0.1, iou_threshold=0.45)
        assert len(results) == 4

    def test_confidences_sorted_descending(self):
        """Returned detections should have the highest confidence first."""
        preds = _make_decoded_preds(B=1, N=20)
        preds[..., 4] = torch.linspace(0.5, 0.9, 20)
        preds[..., 5] = 0.9
        results = non_max_suppression(preds, conf_threshold=0.1, iou_threshold=0.45)
        if results[0] is not None and len(results[0]) > 1:
            confs = results[0][:, 4]
            assert (np.diff(confs) <= 0).all(), "Confidences should be non-increasing"
