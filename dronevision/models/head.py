"""
DroneHead — Anchor-based detection head for 3 scales, 3 anchors/scale.

Each anchor predicts 6 values per spatial location:
    [tx, ty, tw, th, obj_conf, cls_conf]
    ↑box offset ↑box size  ↑objectness ↑class prob (drone)

For training: returns raw (unactivated) predictions.
For inference: caller applies sigmoid and decodes via utils.anchors.decode_predictions().

Output shape per scale:
    (B, num_anchors, H, W, 5 + num_classes)
    = (B, 3, H, W, 6)  for num_classes=1

The head uses a simple 2-stage design:
    CBS(3×3) → Conv(1×1)   — one refinement conv before the final predictor.

This is intentionally simpler than a decoupled head (separate branches for
box/cls) to keep Phase 1 architecture minimal. A decoupled head can be
introduced in Phase 2 if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dronevision.models.blocks import CBS
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionScale(nn.Module):
    """
    Single-scale detection head module.

    Args:
        in_ch:      Input feature map channels.
        num_anchors: Number of anchors at this scale (default: 3).
        num_classes: Number of object classes (default: 1 for drone-only).
    """

    def __init__(
        self,
        in_ch: int,
        num_anchors: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)  # 3 × 6 = 18 for nc=1
        self.conv = CBS(in_ch, in_ch, k=3, s=1)
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self._init_biases()

    def _init_biases(self) -> None:
        """
        Initialize objectness bias to log(prior_prob / (1 - prior_prob)).

        A prior objectness probability of 0.01 ensures the model starts by
        predicting almost no objects, which is appropriate given the extreme
        class imbalance (very few drones vs many background locations).
        """
        prior_prob = 0.01
        import math
        bias_val = math.log(prior_prob / (1 - prior_prob))
        b = self.pred.bias.data
        # Bias layout: [t,x, ty, tw, th, obj, cls] × num_anchors (flattened over anchors)
        # Set objectness bias (index 4 in each 6-element group)
        nc5 = 5 + self.num_classes
        for a in range(self.num_anchors):
            b[a * nc5 + 4] = bias_val  # objectness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W) feature map.

        Returns:
            (B, num_anchors, H, W, 5+nc) raw prediction tensor.
        """
        B, _, H, W = x.shape
        out = self.pred(self.conv(x))                         # (B, 18, H, W)
        nc5 = 5 + self.num_classes
        # Reshape: (B, 18, H, W) → (B, 3, H, W, 6)
        out = out.view(B, self.num_anchors, nc5, H, W)        # (B, 3, 6, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()         # (B, 3, H, W, 6)
        return out


class DroneHead(nn.Module):
    """
    Multi-scale anchor-based detection head.

    Args:
        in_channels: Tuple of 3 channel counts matching neck outputs (N3, N4, N5).
        num_classes: Number of object classes (default: 1).
        num_anchors: Anchors per scale (default: 3).
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 1,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.scale0 = DetectionScale(in_channels[0], num_anchors, num_classes)  # N3
        self.scale1 = DetectionScale(in_channels[1], num_anchors, num_classes)  # N4
        self.scale2 = DetectionScale(in_channels[2], num_anchors, num_classes)  # N5

        logger.info(
            "DroneHead initialized | anchors_per_scale=%d | nc=%d | "
            "output_ch_per_scale=%d",
            num_anchors,
            num_classes,
            num_anchors * (5 + num_classes),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            features: (N3, N4, N5) from neck.

        Returns:
            List of 3 prediction tensors, each (B, num_anchors, H_i, W_i, 5+nc).
            [scale0_pred, scale1_pred, scale2_pred] where:
              scale0 → 80×80 (small drones)
              scale1 → 40×40 (medium drones)
              scale2 → 20×20 (large drones)
        """
        n3, n4, n5 = features
        return [
            self.scale0(n3),
            self.scale1(n4),
            self.scale2(n5),
        ]
