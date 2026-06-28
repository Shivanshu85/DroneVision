"""
DronePredictor — End-to-end inference for single images and batches.

Pipeline:
    Image → Letterbox → ToTensor → DroneDetector → NMS → Rescale → (boxes, count)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dronevision.data.transforms import Letterbox, ToTensor, rescale_boxes
from dronevision.inference.nms import non_max_suppression
from dronevision.models.detector import DroneDetector
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class DronePredictor:
    """
    High-level inference interface for DroneDetector.

    Args:
        model:          Loaded DroneDetector in eval mode.
        cfg:            Configuration dict (for thresholds).
        device:         Compute device.
    """

    def __init__(
        self,
        model: DroneDetector,
        cfg: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        img_size = cfg["model"]["image_size"]
        self.letterbox = Letterbox(target_size=img_size)
        self.to_tensor = ToTensor(normalize=True)

        self.conf_threshold = cfg["inference"]["conf_threshold"]
        self.iou_threshold = cfg["inference"]["iou_threshold"]

        logger.info(
            "DronePredictor ready | img_size=%d | conf=%.2f | iou=%.2f | device=%s",
            img_size, self.conf_threshold, self.iou_threshold, self.device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfg: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> "DronePredictor":
        """
        Create a DronePredictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            cfg:             Configuration dict matching the checkpoint.
            device:          Target device.

        Returns:
            DronePredictor instance ready for inference.
        """
        model, _ = DroneDetector.load_checkpoint(checkpoint_path, cfg, device)
        return cls(model, cfg, device)

    def predict_image(
        self,
        image: np.ndarray | str | Path,
    ) -> dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: BGR numpy array (HWC uint8) or path to an image file.

        Returns:
            Dict with keys:
              - "boxes":       (M, 4) float32 array [x1,y1,x2,y2] normalized [0,1].
              - "confidences": (M,) float32 confidence scores.
              - "drone_count": int — number of detected drones.
              - "meta":        Letterbox meta dict for coordinate rescaling.
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image}")

        # Preprocess
        img_lb, _, meta = self.letterbox(image.copy(), None)
        img_t = torch.from_numpy(self.to_tensor(img_lb)).unsqueeze(0)
        img_t = img_t.to(self.device)

        # Inference
        with torch.no_grad():
            decoded = self.model(img_t)  # (1, N, 5+nc)

        # NMS
        dets = non_max_suppression(
            decoded,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        det = dets[0]  # (M, 6) or None

        if det is None or len(det) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "confidences": np.zeros(0, dtype=np.float32),
                "drone_count": 0,
                "meta": meta,
            }

        boxes_norm = det[:, :4]   # (M, 4) normalized xyxy in letterboxed space
        confs = det[:, 4]         # (M,)

        # Convert back to cxcywh for rescaling, then back to xyxy
        cxcywh = np.stack([
            (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2,
            (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2,
            boxes_norm[:, 2] - boxes_norm[:, 0],
            boxes_norm[:, 3] - boxes_norm[:, 1],
        ], axis=1)
        cxcywh_orig = rescale_boxes(cxcywh, meta)
        # Convert to xyxy
        boxes_orig = np.stack([
            cxcywh_orig[:, 0] - cxcywh_orig[:, 2] / 2,
            cxcywh_orig[:, 1] - cxcywh_orig[:, 3] / 2,
            cxcywh_orig[:, 0] + cxcywh_orig[:, 2] / 2,
            cxcywh_orig[:, 1] + cxcywh_orig[:, 3] / 2,
        ], axis=1)
        boxes_orig = np.clip(boxes_orig, 0.0, 1.0)

        return {
            "boxes": boxes_orig.astype(np.float32),
            "confidences": confs.astype(np.float32),
            "drone_count": len(boxes_orig),
            "meta": meta,
        }

    def predict_batch(
        self,
        image_paths: list[str | Path],
    ) -> list[dict[str, Any]]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of result dicts (same format as predict_image()).
        """
        results = []
        for path in image_paths:
            result = self.predict_image(path)
            result["path"] = str(path)
            results.append(result)
        return results
