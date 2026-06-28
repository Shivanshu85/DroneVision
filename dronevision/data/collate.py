"""
Custom collate function for DroneVision DataLoader.

The default PyTorch collate cannot handle a batch where each sample has a
different number of bounding boxes. This collate function handles variable-length
label tensors by prepending the batch index to each box, then concatenating
all boxes into a single (N_total, 6) tensor.

The batch index allows the loss function to correctly associate each box
with its source image in the batch.
"""

from __future__ import annotations

import torch


def drone_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Collate a batch of (image, labels, path) samples.

    Args:
        batch: List of tuples returned by DroneDataset.__getitem__:
               - image:   (3, H, W) float32 tensor
               - labels:  (N_i, 5) float32 tensor [cls, cx, cy, w, h] or (0, 5)
               - path:    str

    Returns:
        images:  (B, 3, H, W) stacked tensor.
        targets: (N_total, 6) tensor where each row is
                 [batch_idx, cls, cx, cy, w, h].
                 Empty tensor (0, 6) if no GT boxes exist in the batch.
        paths:   list of B path strings.
    """
    images, labels_list, paths = zip(*batch)

    # Stack images into batch
    images_tensor = torch.stack(images, dim=0)

    # Build targets with batch indices
    target_parts: list[torch.Tensor] = []
    for batch_idx, labels in enumerate(labels_list):
        if len(labels) > 0:
            # Prepend batch index column: (N, 5) → (N, 6)
            idx_col = torch.full(
                (len(labels), 1),
                fill_value=float(batch_idx),
                dtype=torch.float32,
            )
            target_parts.append(torch.cat([idx_col, labels], dim=1))

    if target_parts:
        targets = torch.cat(target_parts, dim=0)  # (N_total, 6)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return images_tensor, targets, list(paths)
