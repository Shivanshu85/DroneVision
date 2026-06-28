"""
DroneDataset — PyTorch Dataset for drone detection training and evaluation.

Expects YOLO-format labels (after conversion from Pascal VOC via converter.py):
    datasets/
        images/
            train/ val/ test/
        labels/
            train/ val/ test/

Label file format (one line per drone):
    0 cx cy w h    (all normalized to [0.0, 1.0])

Background images (no drones) have empty label files or no label file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dronevision.data.augmentation import DroneAugmentation
from dronevision.data.transforms import Letterbox, ToTensor
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image extensions
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


class DroneDataset(Dataset):
    """
    Dataset for single-class drone detection.

    Args:
        image_dir:   Path to the image directory (e.g. datasets/images/train).
        label_dir:   Path to the label directory (e.g. datasets/labels/train).
        img_size:    Target image size (square) after letterbox.
        augment:     If True, apply the augmentation pipeline.
        aug_cfg:     Augmentation configuration dict (from YAML).
        is_training: If True, enables augmentation. If False, applies only
                     letterbox + ToTensor (for validation/test).
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        img_size: int = 640,
        augment: bool = False,
        aug_cfg: dict | None = None,
        is_training: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment and is_training
        self.is_training = is_training

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir.resolve()}\n"
                "Run 'python scripts/convert_voc_to_yolo.py' first."
            )

        if not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir.resolve()}\n"
                "Run 'python scripts/convert_voc_to_yolo.py' first."
            )

        # Collect all valid image paths
        self.image_paths: list[Path] = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.image_dir.resolve()} "
                f"with extensions {_IMAGE_EXTS}"
            )

        logger.info(
            "DroneDataset [%s]: %d images found",
            "train" if is_training else "val/test",
            len(self.image_paths),
        )

        # Transforms
        self.letterbox = Letterbox(target_size=img_size)
        self.to_tensor = ToTensor(normalize=True)

        # Augmentation (training only)
        self.augmentation: DroneAugmentation | None = None
        if self.augment:
            self.augmentation = DroneAugmentation(aug_cfg or {}, img_size=img_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load, preprocess, and optionally augment one sample.

        Args:
            idx: Sample index.

        Returns:
            image:   (3, H, W) float32 tensor, normalized to [0, 1].
            targets: (N, 5) float32 tensor [cls, cx, cy, w, h] normalized,
                     or empty tensor (0, 5) for background images.
            path:    Absolute path string of the source image.
        """
        image, boxes, path = self.load_raw(idx)

        # Letterbox resize
        image, boxes, _ = self.letterbox(image, boxes)

        # Apply augmentation (training only)
        if self.augmentation is not None:
            image, boxes = self.augmentation.apply(image, boxes, dataset=self)

        # Convert to tensor
        image_tensor = torch.from_numpy(self.to_tensor(image))

        # Convert boxes to tensor
        if boxes is not None and len(boxes) > 0:
            targets = torch.from_numpy(boxes.astype(np.float32))
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return image_tensor, targets, str(path)

    def load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray | None, Path]:
        """
        Load the raw (un-preprocessed) image and label for sample idx.

        Used internally by augmentation functions (Mosaic, MixUp) to fetch
        additional random samples without going through the full pipeline.

        Args:
            idx: Sample index.

        Returns:
            (image_bgr, boxes, image_path)
            boxes: (N, 5) [cls, cx, cy, w, h] normalized or None.
        """
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read image: %s — returning blank", img_path)
            image = np.full((480, 640, 3), 114, dtype=np.uint8)

        # Load label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = self._load_label(label_path)

        return image, boxes, img_path

    @staticmethod
    def _load_label(label_path: Path) -> np.ndarray:
        """
        Parse a YOLO label file.

        Args:
            label_path: Path to the .txt label file.

        Returns:
            (N, 5) float32 array [cls, cx, cy, w, h] or
            (0, 5) empty array if the file is missing or empty.
        """
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        # Fast-path: empty file → background image (avoids NumPy UserWarning)
        if label_path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        try:
            data = np.loadtxt(str(label_path), dtype=np.float32, ndmin=2)
        except Exception:  # noqa: BLE001
            return np.zeros((0, 5), dtype=np.float32)

        if data.size == 0 or data.ndim != 2 or data.shape[1] != 5:
            return np.zeros((0, 5), dtype=np.float32)

        return data

    def get_label_stats(self) -> dict[str, Any]:
        """
        Compute summary statistics over the full dataset labels.

        Returns:
            Dict with keys: total_images, total_boxes, background_images,
                            avg_boxes_per_image, box_wh_array.
        """
        total_boxes = 0
        background = 0
        all_wh: list[list[float]] = []

        for img_path in self.image_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            boxes = self._load_label(label_path)
            if len(boxes) == 0:
                background += 1
            else:
                total_boxes += len(boxes)
                all_wh.extend(boxes[:, 3:5].tolist())

        n = len(self.image_paths)
        return {
            "total_images": n,
            "total_boxes": total_boxes,
            "background_images": background,
            "avg_boxes_per_image": total_boxes / max(n - background, 1),
            "box_wh_array": np.array(all_wh, dtype=np.float32),
        }
