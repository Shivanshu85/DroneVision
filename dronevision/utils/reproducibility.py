"""
Reproducibility utilities for DroneVision.

Call set_seed() at the start of every training run to ensure that
experiments are reproducible given the same config and seed value.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from dronevision.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Seed all random number generators for reproducible training.

    Affects:
        - Python's random module
        - NumPy
        - PyTorch CPU and CUDA RNGs
        - CuDNN deterministic mode

    Args:
        seed: Integer seed value (default: 42).

    Note:
        CuDNN deterministic mode may reduce performance slightly.
        Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment if
        torch.use_deterministic_algorithms(True) is desired.
    """
    logger.info("Setting global random seed: %d", seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable auto-tuner for reproducibility


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Select the best available compute device.

    Priority order (as specified in TRD.md):
        1. CUDA
        2. MPS (Apple Silicon)
        3. CPU

    Args:
        prefer_cuda: If True (default), check for CUDA first.

    Returns:
        torch.device for the selected backend.

    Raises:
        RuntimeError: If training is attempted on CPU when prefer_cuda=True
                      and CUDA is unavailable (logs a warning instead of raising).
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using CUDA device: %s (VRAM: %.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
        return device

    logger.warning(
        "GPU not available — falling back to CPU. "
        "Training will be significantly slower. "
        "Verify your CUDA installation if GPU is expected."
    )
    return torch.device("cpu")
