"""
Anchor generation utilities for DroneVision.

Anchors define the prior box shapes used by the detection head.
They are stored as pixel-space (w, h) pairs for each detection scale.

The detection head decodes predictions as:
    bx = (sigmoid(tx)*2 - 0.5 + cx_cell) * stride
    by = (sigmoid(ty)*2 - 0.5 + cy_cell) * stride
    bw = (sigmoid(tw)*2)^2 * anchor_w
    bh = (sigmoid(th)*2)^2 * anchor_h

This encoding keeps gradients smooth and prevents exp() overflow.
"""

from __future__ import annotations

import torch
import numpy as np


def build_anchor_tensor(
    anchor_cfg: list[list[list[int]]],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Convert the config anchor list to a (num_scales, num_anchors, 2) tensor.

    Args:
        anchor_cfg: Config list, e.g.
            [[[6,6],[10,10],[16,16]], [[24,24],[38,38],[56,56]], ...]
        device: Target device.

    Returns:
        (3, 3, 2) float32 tensor: anchors[scale][anchor] = [w, h] in pixels.
    """
    return torch.tensor(anchor_cfg, dtype=torch.float32, device=device)


def make_grid(
    grid_h: int,
    grid_w: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Create a 2D grid of (x, y) cell origin coordinates.

    Args:
        grid_h: Number of rows (height) in the feature map.
        grid_w: Number of columns (width) in the feature map.
        device: Target device.

    Returns:
        (grid_h, grid_w, 2) float32 tensor: grid[j][i] = [i, j] (x=col, y=row).
    """
    ys = torch.arange(grid_h, dtype=torch.float32, device=device)
    xs = torch.arange(grid_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
    return torch.stack([grid_x, grid_y], dim=-1)             # (H, W, 2)


def decode_predictions(
    raw: torch.Tensor,
    anchors: torch.Tensor,
    stride: int,
    img_size: int,
) -> torch.Tensor:
    """
    Decode raw head output to normalized [cx, cy, w, h, obj, cls...].

    Args:
        raw:      (B, num_anchors, H, W, 5+nc) raw predictions.
        anchors:  (num_anchors, 2) anchor [w, h] in pixel space.
        stride:   Detection scale stride (e.g. 8, 16, 32).
        img_size: Square image size (e.g. 640).

    Returns:
        (B, num_anchors, H, W, 5+nc) decoded predictions, all in [0,1] except
        obj/cls which are sigmoid-activated probabilities.
    """
    B, na, H, W, nc5 = raw.shape
    device = raw.device

    grid = make_grid(H, W, device=device)  # (H, W, 2)

    out = raw.clone()

    # xy: center of box relative to image, normalized
    out[..., 0:2] = (torch.sigmoid(raw[..., 0:2]) * 2 - 0.5 + grid) * stride / img_size

    # wh: box dimensions normalized by image size
    anchor_tensor = anchors.view(1, na, 1, 1, 2).to(device)
    out[..., 2:4] = (torch.sigmoid(raw[..., 2:4]) * 2) ** 2 * anchor_tensor / img_size

    # obj & class: sigmoid activations
    out[..., 4:] = torch.sigmoid(raw[..., 4:])

    return out


def suggest_anchors_kmeans(
    box_wh: np.ndarray,
    k: int = 9,
    img_size: int = 640,
    n_iter: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Suggest k anchor sizes using k-means clustering on GT box dimensions.

    Args:
        box_wh:   (N, 2) array of [w, h] values (pixel space, img_size scale).
        k:        Number of clusters (typically 9 = 3 scales × 3 anchors).
        img_size: Target image size for normalization display.
        n_iter:   Maximum k-means iterations.
        seed:     Random seed.

    Returns:
        (k, 2) array of sorted anchor [w, h] pairs (sorted by area).
    """
    np.random.seed(seed)
    n = len(box_wh)

    if n < k:
        raise ValueError(f"Need at least {k} boxes to cluster, got {n}")

    # Normalize to [0,1] for clustering stability
    wh = box_wh / img_size

    # Initialize centroids randomly
    idx = np.random.choice(n, k, replace=False)
    centroids = wh[idx]

    for _ in range(n_iter):
        # Assign each box to nearest centroid (using 1-IoU distance)
        dists = _wh_iou_distance(wh, centroids)       # (N, k)
        assignments = dists.argmin(axis=1)             # (N,)

        # Update centroids
        new_centroids = np.array([
            wh[assignments == j].mean(axis=0) if (assignments == j).any() else centroids[j]
            for j in range(k)
        ])

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Scale back to pixel space and sort by area
    anchors_px = centroids * img_size
    areas = anchors_px[:, 0] * anchors_px[:, 1]
    return anchors_px[np.argsort(areas)]


def _wh_iou_distance(
    wh1: np.ndarray,
    wh2: np.ndarray,
) -> np.ndarray:
    """
    Compute 1 - IoU between all pairs of (w,h) boxes anchored at origin.

    Args:
        wh1: (N, 2) normalized width-height pairs.
        wh2: (K, 2) normalized width-height pairs (centroids).

    Returns:
        (N, K) distance matrix (1 - IoU).
    """
    inter = np.minimum(wh1[:, None, :], wh2[None, :, :]).prod(axis=2)  # (N, K)
    area1 = wh1.prod(axis=1)[:, None]
    area2 = wh2.prod(axis=1)[None, :]
    union = area1 + area2 - inter + 1e-7
    return 1.0 - inter / union
