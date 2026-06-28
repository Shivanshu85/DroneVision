#!/usr/bin/env python3
"""
Training entry point for DroneVision.

Usage:
    # Development run (fast pipeline verification)
    python scripts/train.py --config configs/dev.yaml

    # Phase 1 training
    python scripts/train.py --config configs/phase1.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/phase1.yaml --resume runs/phase1/last.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.engine.trainer import DroneTrainer
from dronevision.utils.config import load_config
from dronevision.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DroneVision detector")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to YAML config file (configs/dev.yaml or configs/phase1.yaml)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of training epochs from config.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size from config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        logger.info("Epoch override: %d", args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        logger.info("Batch size override: %d", args.batch_size)

    logger.info(
        "Starting training | config=%s | epochs=%d | batch=%d",
        args.config, cfg["train"]["epochs"], cfg["train"]["batch_size"],
    )

    trainer = DroneTrainer(cfg, resume_path=args.resume)
    final_metrics = trainer.train()

    print("\n" + "=" * 50)
    print("  Training Complete")
    print("=" * 50)
    for k, v in final_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
