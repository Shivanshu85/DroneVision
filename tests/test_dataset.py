"""Tests for DroneDataset and drone_collate_fn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dronevision.data.dataset import DroneDataset
from dronevision.data.collate import drone_collate_fn


class TestDroneDataset:
    def test_loads_images_from_synthetic_dir(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
            augment=False,
            is_training=False,
        )
        assert len(ds) == 3

    def test_getitem_returns_correct_types(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        img, targets, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert isinstance(path, str)

    def test_image_tensor_shape(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        img, _, _ = ds[0]
        assert img.shape == (3, 416, 416)
        assert img.dtype == torch.float32

    def test_image_pixel_range(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        img, _, _ = ds[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_labeled_image_has_boxes(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        # img_0000 has 2 boxes, img_0001 has 1 box, img_0002 is background
        counts = []
        for i in range(3):
            _, targets, path = ds[i]
            counts.append(len(targets))
        assert sorted(counts) == [0, 1, 2]

    def test_background_image_has_empty_targets(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        for i in range(3):
            _, targets, path = ds[i]
            if "img_0002" in path:
                assert targets.shape == (0, 5)

    def test_box_values_normalized(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        for i in range(3):
            _, targets, _ = ds[i]
            if len(targets) > 0:
                # class, cx, cy, w, h all in [0,1]
                assert targets[:, 1:].min() >= 0.0
                assert targets[:, 1:].max() <= 1.0
                # All class IDs must be 0
                assert (targets[:, 0] == 0).all()

    def test_load_raw_returns_numpy(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        img, boxes, path = ds.load_raw(0)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_missing_label_dir_raises(self, tmp_path: Path):
        import cv2
        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "a.png"), np.zeros((100, 100, 3), dtype=np.uint8))

        with pytest.raises(FileNotFoundError):
            DroneDataset(
                image_dir=img_dir,
                label_dir=str(tmp_path / "labels" / "train"),  # doesn't exist
                img_size=416,
            )


class TestDroneCollate:
    def test_collate_stacks_images(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        batch = [ds[i] for i in range(3)]
        images, targets, paths = drone_collate_fn(batch)
        assert images.shape == (3, 3, 416, 416)

    def test_collate_targets_have_batch_index(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        batch = [ds[i] for i in range(3)]
        images, targets, paths = drone_collate_fn(batch)
        # targets: (N_total, 6) [batch_idx, cls, cx, cy, w, h]
        assert targets.ndim == 2
        assert targets.shape[1] == 6

    def test_collate_batch_indices_correct(self, synthetic_dataset_dir: Path):
        ds = DroneDataset(
            image_dir=synthetic_dataset_dir / "images" / "train",
            label_dir=str(synthetic_dataset_dir / "labels" / "train"),
            img_size=416,
        )
        batch = [ds[i] for i in range(3)]
        _, targets, _ = drone_collate_fn(batch)
        if len(targets) > 0:
            unique_idxs = targets[:, 0].unique().long().tolist()
            for idx in unique_idxs:
                assert idx in [0, 1, 2]

    def test_all_background_batch_empty_targets(self):
        """A batch of all-background images should produce (0,6) targets."""
        empty_img = torch.zeros(3, 416, 416)
        empty_tgt = torch.zeros(0, 5)
        batch = [(empty_img, empty_tgt, "a"), (empty_img, empty_tgt, "b")]
        _, targets, _ = drone_collate_fn(batch)
        assert targets.shape == (0, 6)
