"""Tests for data augmentation and transforms."""

from __future__ import annotations

import numpy as np
import pytest

from dronevision.data.transforms import Letterbox, ToTensor, rescale_boxes
from dronevision.data.augmentation import DroneAugmentation


class TestLetterbox:
    def test_square_output_shape(self, random_image_416):
        lb = Letterbox(target_size=416)
        out, _, _ = lb(random_image_416, None)
        assert out.shape == (416, 416, 3)

    def test_non_square_input_padded_correctly(self):
        lb = Letterbox(target_size=640)
        img = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        out, _, meta = lb(img, None)
        assert out.shape == (640, 640, 3)
        assert meta["ratio"] > 0

    def test_boxes_remain_in_unit_range(self, random_image_416, random_boxes_5):
        lb = Letterbox(target_size=416)
        _, boxes_out, _ = lb(random_image_416, random_boxes_5)
        assert boxes_out is not None
        assert boxes_out[:, 1:].min() >= 0.0
        assert boxes_out[:, 1:].max() <= 1.0

    def test_empty_boxes_handled(self, random_image_416):
        lb = Letterbox(target_size=416)
        boxes = np.zeros((0, 5), dtype=np.float32)
        out, boxes_out, meta = lb(random_image_416, boxes)
        assert out.shape[0] == 416
        assert boxes_out is not None and len(boxes_out) == 0

    def test_no_boxes_none_handled(self, random_image_416):
        lb = Letterbox(target_size=416)
        out, boxes_out, meta = lb(random_image_416, None)
        assert out.shape[0] == 416
        assert boxes_out is None


class TestToTensor:
    def test_output_shape_chw(self, random_image_416):
        tt = ToTensor(normalize=True)
        out = tt(random_image_416)
        assert out.shape == (3, 416, 416)

    def test_normalized_range(self, random_image_416):
        tt = ToTensor(normalize=True)
        out = tt(random_image_416)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_dtype_float32(self, random_image_416):
        tt = ToTensor(normalize=True)
        out = tt(random_image_416)
        assert out.dtype == np.float32


class TestRescaleBoxes:
    def test_rescale_is_inverse_of_letterbox(self):
        """Rescaling boxes through letterbox and back should approximately recover original."""
        lb = Letterbox(target_size=640)
        img = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        boxes_orig = np.array([[0.3, 0.4, 0.1, 0.1]], dtype=np.float32)  # cxcywh
        _, boxes_lb, meta = lb(img, np.array([[0.0, 0.3, 0.4, 0.1, 0.1]]))
        boxes_rescaled = rescale_boxes(boxes_lb[:, 1:], meta)
        np.testing.assert_allclose(boxes_rescaled[0], boxes_orig[0], atol=0.02)


class TestDroneAugmentation:
    def _make_aug(self, **overrides) -> DroneAugmentation:
        cfg = {
            "horizontal_flip": 1.0,  # always flip for determinism tests
            "vertical_flip": 0.0,
            "color_jitter": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "random_scale": 0.0,
            "gaussian_blur": 0.0,
        }
        cfg.update(overrides)
        return DroneAugmentation(cfg, img_size=416)

    def test_hflip_output_shape(self, random_image_416, random_boxes_5):
        aug = self._make_aug(horizontal_flip=1.0)
        out_img, out_boxes = aug.apply(random_image_416, random_boxes_5, dataset=None)
        assert out_img.shape == random_image_416.shape

    def test_hflip_mirrors_cx(self, random_image_416, random_boxes_5):
        aug = self._make_aug(horizontal_flip=1.0)
        out_img, out_boxes = aug.apply(random_image_416, random_boxes_5, dataset=None)
        expected_cx = 1.0 - random_boxes_5[:, 1]
        np.testing.assert_allclose(out_boxes[:, 1], expected_cx, atol=1e-5)

    def test_color_jitter_output_shape(self, random_image_416, random_boxes_5):
        aug = self._make_aug(horizontal_flip=0.0, color_jitter=1.0)
        out_img, _ = aug.apply(random_image_416, random_boxes_5, dataset=None)
        assert out_img.shape == random_image_416.shape

    def test_output_pixel_range_uint8(self, random_image_416, random_boxes_5):
        aug = self._make_aug(color_jitter=1.0, gaussian_blur=1.0)
        out_img, _ = aug.apply(random_image_416, random_boxes_5, dataset=None)
        assert out_img.dtype == np.uint8

    def test_empty_boxes_preserved(self, random_image_416):
        aug = self._make_aug(horizontal_flip=1.0)
        empty = np.zeros((0, 5), dtype=np.float32)
        out_img, out_boxes = aug.apply(random_image_416, empty, dataset=None)
        assert len(out_boxes) == 0
