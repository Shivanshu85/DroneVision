"""
DroneNeck — Lightweight Feature Pyramid Network (FPN) for drone detection.

Phase 1 uses a top-down FPN only (no PAN bottom-up path).
PAN is deferred to Phase 2 once baseline performance is established.

Why FPN for drone detection?
    Drones appear at vastly different scales depending on altitude and distance.
    FPN merges high-level semantic features (from deep layers) with high-resolution
    spatial features (from shallow layers) via a top-down pathway.

    Without FPN: P3 (small drone scale) would lack semantic context.
                 P5 (large drone scale) would lack spatial precision.
    With FPN: each scale gets both semantic richness and spatial detail.

Architecture:

    Input:  P3=(B,128,H/8,W/8)  P4=(B,256,H/16,W/16)  P5=(B,256,H/32,W/32)

    Lateral projections (reduce P5, P4, P3 channels):
        lat_P5 = CBS(256→128)(P5)                          → (B,128,H/32,W/32)
        lat_P4 = CBS(256→128)(P4)                          → (B,128,H/16,W/16)
        lat_P3 = CBS(128→64)(P3)                           → (B,64, H/8, W/8)

    Top-down merging (upsample + concat + refine):
        N5 = lat_P5                                        → (B,128,H/32,W/32)
        N4 = CBS(256→128)(cat[lat_P4, up2(N5)])            → (B,128,H/16,W/16)
        N3 = CBS(192→64)(cat[lat_P3, up2(N4)])             → (B,64, H/8, W/8)

    Output: (N3, N4, N5)
        N3: (B, 64,  H/8,  W/8)   — small drones
        N4: (B, 128, H/16, W/16)  — medium drones
        N5: (B, 128, H/32, W/32)  — large drones
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dronevision.models.blocks import CBS
from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class DroneNeck(nn.Module):
    """
    Lightweight Feature Pyramid Network neck.

    Args:
        in_channels: Tuple of 3 channel counts (P3_ch, P4_ch, P5_ch)
                     matching the backbone output channels.
        out_channels: Tuple of 3 output channel counts (N3_ch, N4_ch, N5_ch).
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (128, 256, 256),
        out_channels: tuple[int, int, int] = (64, 128, 128),
    ) -> None:
        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels
        n3_ch, n4_ch, n5_ch = out_channels

        # Lateral projections: compress channels before merging
        self.lat_p5 = CBS(p5_ch, n5_ch, k=1, s=1)     # 256 → 128
        self.lat_p4 = CBS(p4_ch, n4_ch, k=1, s=1)     # 256 → 128
        self.lat_p3 = CBS(p3_ch, n3_ch, k=1, s=1)     # 128 → 64

        # Top-down fusion layers
        # P4 + upsampled(N5): concat channels = n4_ch + n5_ch
        self.merge_p4 = CBS(n4_ch + n5_ch, n4_ch, k=3, s=1)

        # P3 + upsampled(N4): concat channels = n3_ch + n4_ch
        self.merge_p3 = CBS(n3_ch + n4_ch, n3_ch, k=3, s=1)

        self._log_info(in_channels, out_channels)

    def _log_info(
        self,
        in_channels: tuple[int, int, int],
        out_channels: tuple[int, int, int],
    ) -> None:
        params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DroneNeck (FPN) initialized | in=%s → out=%s | params=%.2fM",
            in_channels, out_channels, params / 1e6,
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (P3, P4, P5) from backbone.
                P3: (B, 128, H/8,  W/8)
                P4: (B, 256, H/16, W/16)
                P5: (B, 256, H/32, W/32)

        Returns:
            (N3, N4, N5) detection-ready feature maps:
                N3: (B, 64,  H/8,  W/8)
                N4: (B, 128, H/16, W/16)
                N5: (B, 128, H/32, W/32)
        """
        p3, p4, p5 = features

        # Lateral connections
        n5 = self.lat_p5(p5)   # (B, 128, H/32, W/32)
        lat4 = self.lat_p4(p4) # (B, 128, H/16, W/16)
        lat3 = self.lat_p3(p3) # (B, 64,  H/8,  W/8)

        # Top-down: P4 level
        n5_up = F.interpolate(n5, size=lat4.shape[2:], mode="nearest")
        n4 = self.merge_p4(torch.cat([lat4, n5_up], dim=1))  # (B, 128, H/16, W/16)

        # Top-down: P3 level
        n4_up = F.interpolate(n4, size=lat3.shape[2:], mode="nearest")
        n3 = self.merge_p3(torch.cat([lat3, n4_up], dim=1))  # (B, 64, H/8, W/8)

        return n3, n4, n5
