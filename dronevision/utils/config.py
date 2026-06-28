"""
YAML configuration loader and validator for DroneVision.

All hyperparameters live exclusively in config files (configs/dev.yaml,
configs/phase1.yaml). No values are hardcoded in the library.

Usage:
    from dronevision.utils.config import load_config
    cfg = load_config("configs/phase1.yaml")
    batch_size = cfg["train"]["batch_size"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dronevision.utils.logger import get_logger

logger = get_logger(__name__)

# Required top-level keys that every config must contain.
_REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "model",
    "data",
    "train",
    "loss",
    "inference",
    "logging",
)

# Required nested keys per section.
_REQUIRED_NESTED: dict[str, tuple[str, ...]] = {
    "model": ("image_size", "num_classes", "num_anchors", "strides", "anchors"),
    "data": ("train", "val"),
    "train": ("epochs", "batch_size", "optimizer", "lr"),
    "loss": ("lambda_box", "lambda_obj", "lambda_cls"),
    "inference": ("conf_threshold", "iou_threshold"),
    "logging": ("mlflow_tracking_uri", "experiment_name"),
}


class ConfigError(ValueError):
    """Raised when a required configuration key is missing or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load and validate a DroneVision YAML configuration file.

    Args:
        path: Absolute or relative path to the YAML config file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If required keys are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    logger.info("Loading config: %s", path.resolve())

    with path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(cfg).__name__}")

    _validate_config(cfg)
    _apply_defaults(cfg)

    logger.debug("Config loaded successfully with %d top-level keys", len(cfg))
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate presence of required keys."""
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ConfigError(f"Missing required top-level config key: '{key}'")

    for section, required_keys in _REQUIRED_NESTED.items():
        section_cfg = cfg.get(section, {})
        for key in required_keys:
            if key not in section_cfg:
                raise ConfigError(
                    f"Missing required key '{key}' in config section '{section}'"
                )

    # Validate anchor structure
    anchors = cfg["model"]["anchors"]
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ConfigError("config.model.anchors must be a list of exactly 3 groups")
    for i, group in enumerate(anchors):
        if not isinstance(group, list) or len(group) != 3:
            raise ConfigError(
                f"config.model.anchors[{i}] must be a list of exactly 3 [w, h] pairs"
            )

    # Validate image size
    img_size = cfg["model"]["image_size"]
    if img_size % 32 != 0:
        raise ConfigError(
            f"config.model.image_size must be divisible by 32, got {img_size}"
        )


def _apply_defaults(cfg: dict[str, Any]) -> None:
    """Apply default values for optional config keys."""
    cfg["train"].setdefault("mixed_precision", False)
    cfg["train"].setdefault("warmup_epochs", 3)
    cfg["train"].setdefault("scheduler", "cosine")
    cfg["train"].setdefault("gradient_clip", 10.0)
    cfg["train"].setdefault("seed", 42)
    cfg["train"].setdefault("save_period", 5)
    cfg["data"].setdefault("num_workers", 4)
    cfg["data"].setdefault("pin_memory", True)
    cfg["augmentation"] = cfg.get("augmentation", {})
    cfg["augmentation"].setdefault("enabled", True)
    cfg["logging"].setdefault("log_interval", 10)
