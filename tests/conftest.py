"""
pytest configuration and shared fixtures for DroneVision test suite.

All fixtures use synthetic data (random tensors and images) so that
tests run without the actual DUT Anti-UAV dataset installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ── Config fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dev_cfg() -> dict:
    """Minimal configuration dict mimicking configs/dev.yaml."""
    return {
        "model": {
            "image_size": 416,
            "num_classes": 1,
            "num_anchors": 3,
            "strides": [8, 16, 32],
            "anchors": [
                [[4, 4], [7, 7], [11, 11]],
                [[16, 16], [25, 25], [37, 37]],
                [[52, 52], [72, 72], [104, 104]],
            ],
            "backbone_channels": [32, 64, 128, 256],
            "neck_channels": [64, 128, 128],
        },
        "data": {
            "train": "datasets/images/train",
            "val": "datasets/images/val",
            "label_dir": "datasets/labels",
            "num_workers": 0,
            "pin_memory": False,
        },
        "augmentation": {"enabled": False},
        "train": {
            "epochs": 2,
            "batch_size": 2,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
            "scheduler": "cosine",
            "mixed_precision": False,
            "gradient_clip": 10.0,
            "seed": 42,
            "checkpoint_dir": "runs/test",
            "save_period": 1,
        },
        "loss": {
            "lambda_box": 7.5,
            "lambda_obj": 1.0,
            "lambda_cls": 0.5,
            "anchor_threshold": 4.0,
            "obj_pos_weight": 1.0,
            "cls_pos_weight": 1.0,
        },
        "inference": {
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        },
        "logging": {
            "mlflow_tracking_uri": "mlruns/",
            "experiment_name": "test",
            "log_interval": 1,
        },
    }


# ── Image / tensor fixtures ───────────────────────────────────────────────────

@pytest.fixture
def random_image_416() -> np.ndarray:
    """Random HWC uint8 BGR image at 416×416."""
    return np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)


@pytest.fixture
def random_image_640() -> np.ndarray:
    """Random HWC uint8 BGR image at 640×640."""
    return np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_boxes_5() -> np.ndarray:
    """(5, 5) random YOLO boxes [cls, cx, cy, w, h] normalized."""
    rng = np.random.default_rng(0)
    boxes = np.zeros((5, 5), dtype=np.float32)
    boxes[:, 0] = 0.0                                 # class = drone
    boxes[:, 1] = rng.uniform(0.1, 0.9, 5)           # cx
    boxes[:, 2] = rng.uniform(0.1, 0.9, 5)           # cy
    boxes[:, 3] = rng.uniform(0.02, 0.15, 5)         # w  (small drones)
    boxes[:, 4] = rng.uniform(0.02, 0.15, 5)         # h
    return boxes


@pytest.fixture
def batch_targets() -> torch.Tensor:
    """(8, 6) batch targets [batch_idx, cls, cx, cy, w, h]."""
    t = torch.zeros(8, 6)
    t[:4, 0] = 0  # batch 0
    t[4:, 0] = 1  # batch 1
    t[:, 1] = 0   # class 0
    t[:, 2] = torch.tensor([0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.15])  # cx
    t[:, 3] = torch.tensor([0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6, 0.35])  # cy
    t[:, 4] = 0.05  # w (small drones)
    t[:, 5] = 0.05  # h
    return t


# ── Dataset fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """
    Create a minimal synthetic YOLO dataset in a temp directory.

    Structure:
        tmp/
            images/train/   (3 PNG images)
            labels/train/   (3 YOLO label files — 2 with boxes, 1 empty)
    """
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    import cv2

    for i in range(3):
        # Write a solid-color PNG image
        img = np.full((480, 640, 3), fill_value=i * 60, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:04d}.png"), img)

    # img_0000: 2 drone boxes
    (lbl_dir / "img_0000.txt").write_text(
        "0 0.300 0.400 0.050 0.040\n0 0.700 0.600 0.060 0.050"
    )
    # img_0001: 1 drone box
    (lbl_dir / "img_0001.txt").write_text("0 0.500 0.500 0.080 0.070")
    # img_0002: background (empty label)
    (lbl_dir / "img_0002.txt").write_text("")

    return tmp_path
