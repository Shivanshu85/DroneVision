"""
DroneDetector — Full detector assembling Backbone + Neck + Head.

This is the top-level model class. All training and inference code
interacts with DroneDetector directly, never with the sub-modules separately.

Forward pass:
    Training mode (model.train()):
        Returns list of 3 raw prediction tensors (unactivated).
        The detection loss function applies sigmoid internally.

    Eval mode (model.eval()):
        Returns (B, N_anchors_total, 5+nc) decoded predictions
        where values are in [0,1] (sigmoid-activated).
        Use inference/nms.py to filter and decode to final boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dronevision.models.backbone import DroneBackbone
from dronevision.models.head import DroneHead
from dronevision.models.neck import DroneNeck
from dronevision.utils.anchors import build_anchor_tensor, decode_predictions
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class DroneDetector(nn.Module):
    """
    Complete drone detector: Backbone → Neck → Head.

    Args:
        cfg: Full configuration dict (from load_config()).

    Architecture summary:
        Backbone: DroneBackbone (~2.3M params)
        Neck:     DroneNeck FPN (~0.2M params)
        Head:     DroneHead (3 scales × 3 anchors) (~0.03M params)
        Total:    ~2.5M params
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.num_classes: int = model_cfg["num_classes"]
        self.num_anchors: int = model_cfg["num_anchors"]
        self.img_size: int = model_cfg["image_size"]
        self.strides: list[int] = model_cfg["strides"]

        # Register anchors as buffer (moves with model to GPU)
        anchors_raw = model_cfg["anchors"]  # list[list[list[int]]]
        anchor_tensor = build_anchor_tensor(anchors_raw)  # (3, 3, 2)
        self.register_buffer("anchors", anchor_tensor)  # (3, 3, 2)

        # Sub-modules
        backbone_ch = model_cfg.get("backbone_channels", [32, 64, 128, 256])
        neck_ch = model_cfg.get("neck_channels", [64, 128, 128])

        self.backbone = DroneBackbone(
            in_channels=3,
            channels=backbone_ch,
        )
        self.neck = DroneNeck(
            in_channels=(backbone_ch[2], backbone_ch[3], backbone_ch[3]),
            out_channels=tuple(neck_ch),
        )
        self.head = DroneHead(
            in_channels=tuple(neck_ch),
            num_classes=self.num_classes,
            num_anchors=self.num_anchors,
        )

        self._log_summary()

    def _log_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DroneDetector ready | total params=%.2fM | trainable=%.2fM "
            "| img_size=%d | nc=%d | anchors/scale=%d",
            total / 1e6,
            trainable / 1e6,
            self.img_size,
            self.num_classes,
            self.num_anchors,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor] | torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor.

        Returns:
            Training mode: list of 3 raw prediction tensors, each
                           (B, num_anchors, H_i, W_i, 5+nc).
            Eval mode:     (B, N_total, 5+nc) decoded predictions
                           where N_total = sum(H_i × W_i × num_anchors).
        """
        # Backbone → multi-scale features
        features = self.backbone(x)

        # Neck → FPN-merged features
        fpn_features = self.neck(features)

        # Head → raw predictions
        raw_preds = self.head(fpn_features)

        if self.training:
            return raw_preds

        # Decode for inference
        return self._decode(raw_preds)

    def _decode(self, raw_preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode raw head outputs to normalized [cx, cy, w, h, obj, cls] predictions.

        Args:
            raw_preds: List of 3 raw tensors from DroneHead.

        Returns:
            (B, N_total, 5+nc) decoded predictions, values in [0,1].
        """
        decoded_scales: list[torch.Tensor] = []
        for scale_idx, (pred, stride) in enumerate(zip(raw_preds, self.strides)):
            B, na, H, W, nc5 = pred.shape
            anchor_set = self.anchors[scale_idx]  # (3, 2)
            decoded = decode_predictions(pred, anchor_set, stride, self.img_size)
            # Reshape to (B, na*H*W, 5+nc)
            decoded = decoded.view(B, na * H * W, nc5)
            decoded_scales.append(decoded)
        return torch.cat(decoded_scales, dim=1)  # (B, N_total, 5+nc)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by sub-module."""
        return {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "neck": sum(p.numel() for p in self.neck.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: dict[str, float] | None = None,
        optimizer_state: dict | None = None,
        scaler_state: dict | None = None,
    ) -> None:
        """
        Save model checkpoint with full training state.

        Args:
            path:            Output file path (.pth).
            epoch:           Current epoch number.
            metrics:         Dict of evaluation metrics at this checkpoint.
            optimizer_state: Optimizer state_dict for training resumption.
            scaler_state:    AMP GradScaler state_dict.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "metrics": metrics or {},
            "anchor_buffer": self.anchors.cpu(),
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scaler_state is not None:
            checkpoint["scaler_state_dict"] = scaler_state
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        cfg: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> tuple["DroneDetector", dict]:
        """
        Load a model from a checkpoint file.

        Args:
            path:   Path to .pth checkpoint.
            cfg:    Configuration dict matching the checkpoint's model config.
            device: Target device.

        Returns:
            (model, checkpoint_dict) tuple.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        epoch = checkpoint.get("epoch", 0)
        logger.info("Checkpoint loaded: %s (epoch %d)", path, epoch)
        return model, checkpoint
