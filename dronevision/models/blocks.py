"""
Shared building blocks for DroneVision model architecture.

All blocks are designed to be:
  - Modular: independently testable
  - Replaceable: drop-in substitution in backbone/neck/head
  - Efficient: minimal VRAM usage for 6 GB GPU budget

Block hierarchy:
    CBS (Conv-BatchNorm-SiLU)  ← fundamental unit
    └─ Bottleneck              ← two CBS blocks + residual skip
    └─ SPPF                    ← spatial context at the top of backbone

Design rationale:
    SiLU (Swish) is used over ReLU throughout because:
    - Smooth non-linearity improves gradient flow for small objects.
    - Better convergence on detection tasks vs ReLU (empirically validated
      in YOLO and related works).
    - No additional parameter cost over ReLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CBS(nn.Module):
    """
    Conv-BatchNorm-SiLU (CBS) block — the fundamental building block.

    Input and output spatial dimensions are controlled via kernel size
    and stride. Padding is auto-computed to preserve spatial size when s=1.

    Args:
        in_ch:  Number of input channels.
        out_ch: Number of output channels.
        k:      Kernel size (default: 1).
        s:      Stride (default: 1).
        p:      Padding (default: auto = k//2).
        g:      Groups for depthwise convolution (default: 1).
        act:    If True, apply SiLU. If False, apply no activation (e.g. for
                the final conv before sigmoid).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2  # auto-padding for 'same' spatial output at s=1
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Residual bottleneck block: CBS(1×1) → CBS(3×3) + optional skip connection.

    The hidden channel dimension is reduced by factor `e` to decrease parameters.
    When `shortcut=True` and in_ch == out_ch, a residual connection is added.

    Args:
        in_ch:    Input (and output) channel count.
        out_ch:   Output channel count (can differ from in_ch when shortcut=False).
        shortcut: Enable residual skip connection (requires in_ch == out_ch).
        e:        Channel expansion ratio for hidden dimension (default: 0.5).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int | None = None,
        shortcut: bool = True,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        hidden = int(in_ch * e)
        self.cv1 = CBS(in_ch, hidden, k=1, s=1)
        self.cv2 = CBS(hidden, out_ch, k=3, s=1)
        self.add = shortcut and (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF).

    Applies three sequential max-pool operations of the same kernel size
    and concatenates the results. This is functionally equivalent to SPP
    (with pool sizes k, 2k-1, 3k-1) but significantly faster.

    Position: top of the backbone (P5 scale) to capture global context.
    This helps detect large/close-range drones that span a substantial
    fraction of the image.

    Args:
        in_ch:  Input channel count.
        out_ch: Output channel count.
        k:      MaxPool kernel size (default: 5).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = CBS(in_ch, hidden, k=1, s=1)
        # After 3 sequential pools + original → 4 × hidden channels
        self.cv2 = CBS(hidden * 4, out_ch, k=1, s=1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.mp(x)
        p2 = self.mp(p1)
        p3 = self.mp(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))
