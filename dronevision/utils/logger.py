"""
Structured logging configuration for DroneVision.

Every module should obtain its logger via:
    from dronevision.utils.logger import get_logger
    logger = get_logger(__name__)

No print() calls should appear in library code.
"""

import logging
import sys
from pathlib import Path


_CONFIGURED: bool = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """
    Configure the root logger once for the entire process.

    Args:
        level:    Logging level (e.g. logging.DEBUG, logging.INFO).
        log_file: Optional path to write logs to a file in addition to stdout.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("dronevision")
    root_logger.setLevel(level)
    root_logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'dronevision' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A Logger instance. If configure_logging() has not been called,
        a default INFO-level stdout handler is applied automatically.
    """
    if not _CONFIGURED:
        configure_logging()
    # Ensure the name is always scoped under dronevision
    if not name.startswith("dronevision"):
        name = f"dronevision.{name}"
    return logging.getLogger(name)
