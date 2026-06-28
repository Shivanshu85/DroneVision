from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dronevision.inference.predictor import DronePredictor
from dronevision.utils.config import load_config
from dronevision.utils.reproducibility import get_device

from demo.config import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from demo.utils import draw_custom_detections

class DemoPredictor:
    """
    Wrapper around the existing DroneVision inference pipeline.
    Ensures the model is loaded once and handles RGB <-> BGR conversions.
    """
    def __init__(
        self, 
        checkpoint_path: Path = DEFAULT_CHECKPOINT, 
        config_path: Path = DEFAULT_CONFIG
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.predictor = None
        self.device_name = "CPU"
        self.error_msg = None
        
        # Load the model
        try:
            self._load_model()
        except Exception as e:
            self.error_msg = f"Failed to initialize predictor: {str(e)}"

    def _load_model(self) -> None:
        """Loads config and checkpoint into memory exactly once."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path.resolve()}.\n"
                "Please run training or place the weights in the specified directory."
            )
            
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path.resolve()}"
            )
            
        # Load configuration dict
        cfg = load_config(self.config_path)
        
        # Determine device
        device = get_device(prefer_cuda=True)
        if device.type == "cuda":
            self.device_name = torch.cuda.get_device_name(0)
        elif device.type == "mps":
            self.device_name = "MPS (Apple Silicon)"
        else:
            self.device_name = "CPU"
            
        # Instantiate predictor
        self.predictor = DronePredictor.from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            cfg=cfg,
            device=device
        )

    def predict(self, rgb_image: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
        """
        Runs inference on an RGB image.
        
        Args:
            rgb_image: HWC uint8 RGB numpy array from Gradio.
            
        Returns:
            A tuple of:
              - Annotated RGB image (or None if error)
              - Summary metrics dictionary
              - Error message string (or None if successful)
        """
        if self.error_msg:
            return None, {}, self.error_msg
            
        if rgb_image is None or not isinstance(rgb_image, np.ndarray):
            return None, {}, "Invalid input image."
            
        try:
            # 1. Convert RGB (Gradio format) to BGR (OpenCV/predictor format)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 2. Benchmark inference duration
            t_start = time.perf_counter()
            result = self.predictor.predict_image(bgr_image)
            t_end = time.perf_counter()
            
            inference_time_ms = (t_end - t_start) * 1000
            
            # 3. Extract outputs
            boxes = result["boxes"]            # (M, 4) normalized [x1, y1, x2, y2]
            confidences = result["confidences"]  # (M,) confidence scores
            drone_count = result["drone_count"]
            
            # 4. Generate custom annotated image using only sequential integers
            annotated_bgr = draw_custom_detections(bgr_image, boxes)
            
            # 5. Convert back to RGB for Gradio display
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            # 6. Package summary metrics (exclude model internals as per product requirement)
            summary = {
                "drone_count": drone_count,
                "inference_time_ms": float(inference_time_ms),
                "device": self.device_name,
                "confidences": [float(c) for c in confidences]
            }
            
            return annotated_rgb, summary, None
            
        except Exception as e:
            return None, {}, f"An error occurred during inference: {str(e)}"
