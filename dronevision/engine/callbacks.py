"""
Training callbacks for DroneVision.

Callbacks are invoked by DroneTrainer at specific points in the training loop.
They are kept separate from the trainer to maintain single responsibility.

Available callbacks:
    EarlyStopping:         Stops training if the monitored metric does not improve.
    WarmupCosineScheduler: Linear warmup followed by cosine annealing.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Stop training when a monitored metric has not improved for `patience` epochs.

    Args:
        patience: Number of epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode:     "max" (higher is better, e.g. mAP50) or "min" (lower is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.stop: bool = False

    def __call__(self, metric: float) -> bool:
        """
        Update state and return True if training should stop.

        Args:
            metric: Current epoch metric value.

        Returns:
            True if training should stop.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best = metric
            self.counter = 0
            logger.debug("EarlyStopping: improved to %.4f", metric)
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement (%d/%d)", self.counter, self.patience
            )

        if self.counter >= self.patience:
            logger.info(
                "EarlyStopping triggered after %d epochs without improvement "
                "(best=%.4f, current=%.4f)",
                self.patience, self.best, metric,
            )
            self.stop = True

        return self.stop


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `min_lr`.

    Why warmup?
        At the start of training, large random weights produce large gradients.
        Warmup gradually increases LR to avoid early instability, which is
        especially important for detection models with many anchor assignments.

    Args:
        optimizer:      PyTorch optimizer.
        warmup_epochs:  Number of linear warmup epochs.
        total_epochs:   Total training epochs.
        min_lr_ratio:   Final LR = initial_lr × min_lr_ratio.
        last_epoch:     Epoch to resume from (-1 for fresh start).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → base_lr over warmup_epochs
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing: base_lr → min_lr
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            ) / 2

        return [base_lr * scale for base_lr in self.base_lrs]
