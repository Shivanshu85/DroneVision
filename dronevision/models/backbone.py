"""
DroneBackbone — Small custom feature extractor for drone detection.

Target parameter count: ~2M (within 1M–5M Phase 1 budget).
Designed for 6 GB VRAM on 640×640 inputs.

Architecture (640×640 input → 3 output scales):

    Stage   Output Shape          Stride  Purpose
    ──────────────────────────────────────────────────────
    0       (B, 32,  320, 320)   2       Initial downsampling
    1       (B, 64,  160, 160)   4       Low-level edges
    2       (B, 128, 80,  80)    8  ← P3 Small drone features
    3       (B, 256, 40,  40)    16 ← P4 Medium drone features
    4+SPPF  (B, 256, 20,  20)    32 ← P5 Large drones + global context

P3 (stride 8, 80×80) is the most critical scale for small drone detection.
Drones at 100–500m range can appear as 5–20px in a 640×640 frame.
At P3, each feature cell covers an 8×8 pixel region, giving adequate
resolution to detect even the smallest UAVs.

Approximate parameter count:
    Stage 0:  ~1K
    Stage 1:  ~19K + 21K (Bottleneck) = ~40K
    Stage 2:  ~74K + 165K (×2 Bottlenecks) = ~239K
    Stage 3:  ~295K + 657K (×2 Bottlenecks) = ~952K
    Stage 4:  ~590K + 328K (Bottleneck) + ~164K (SPPF) = ~1082K
    Total:    ≈ 2.3M parameters
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dronevision.models.blocks import CBS, Bottleneck, SPPF
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class DroneBackbone(nn.Module):
    """
    Small custom backbone for drone detection.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        channels:    List of 4 channel widths [c0, c1, c2, c3].
                     Default [32, 64, 128, 256] targets ~2.3M parameters.

    Returns from forward():
        Tuple of 3 feature maps (P3, P4, P5) for FPN neck input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        c0, c1, c2, c3 = channels

        # Stage 0: Initial downsampling — stride 2 → 320×320
        self.stage0 = CBS(in_channels, c0, k=3, s=2)

        # Stage 1: P2 level — stride 4 → 160×160
        self.stage1 = nn.Sequential(
            CBS(c0, c1, k=3, s=2),
            Bottleneck(c1, c1, shortcut=True, e=0.5),
        )

        # Stage 2: P3 level — stride 8 → 80×80 (primary small-drone scale)
        self.stage2 = nn.Sequential(
            CBS(c1, c2, k=3, s=2),
            Bottleneck(c2, c2, shortcut=True, e=0.5),
            Bottleneck(c2, c2, shortcut=True, e=0.5),
        )

        # Stage 3: P4 level — stride 16 → 40×40 (medium drones)
        self.stage3 = nn.Sequential(
            CBS(c2, c3, k=3, s=2),
            Bottleneck(c3, c3, shortcut=True, e=0.5),
            Bottleneck(c3, c3, shortcut=True, e=0.5),
        )

        # Stage 4: P5 level — stride 32 → 20×20 (large drones + global context)
        self.stage4 = nn.Sequential(
            CBS(c3, c3, k=3, s=2),
            Bottleneck(c3, c3, shortcut=True, e=0.5),
        )
        self.sppf = SPPF(c3, c3, k=5)

        self._log_info(channels)

    def _log_info(self, channels: list[int]) -> None:
        params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DroneBackbone initialized | channels=%s | params=%.2fM",
            channels,
            params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image tensor.

        Returns:
            (P3, P4, P5) feature maps at strides 8, 16, 32.
        """
        x = self.stage0(x)   # (B, 32,  H/2,  W/2)
        x = self.stage1(x)   # (B, 64,  H/4,  W/4)
        p3 = self.stage2(x)  # (B, 128, H/8,  W/8)  ← small drones
        p4 = self.stage3(p3) # (B, 256, H/16, W/16) ← medium drones
        p5 = self.sppf(self.stage4(p4))  # (B, 256, H/32, W/32) ← large drones
        return p3, p4, p5
