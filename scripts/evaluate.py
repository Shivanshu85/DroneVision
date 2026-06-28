#!/usr/bin/env python3
"""
Standalone evaluation script for DroneVision.

Evaluates a trained checkpoint on a dataset split and prints mAP50,
mAP50-95, Precision, Recall, F1.

Usage:
    python scripts/evaluate.py \
        --config configs/phase1.yaml \
        --weights runs/phase1/best.pth \
        --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.data.collate import drone_collate_fn
from dronevision.data.dataset import DroneDataset
from dronevision.engine.evaluator import DroneEvaluator
from dronevision.inference.nms import non_max_suppression
from dronevision.models.detector import DroneDetector
from dronevision.utils.config import load_config
from dronevision.utils.logger import configure_logging, get_logger
from dronevision.utils.reproducibility import get_device

configure_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DroneVision checkpoint")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to .pth checkpoint file.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to save metrics JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = get_device(prefer_cuda=True)
    model, checkpoint = DroneDetector.load_checkpoint(args.weights, cfg, device)

    label_dir = Path(cfg["data"].get("label_dir", "datasets/labels")) / args.split
    img_dir = Path(cfg["data"][args.split])

    ds = DroneDataset(
        image_dir=img_dir,
        label_dir=str(label_dir),
        img_size=cfg["model"]["image_size"],
        augment=False,
        is_training=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=drone_collate_fn,
    )

    evaluator = DroneEvaluator(num_classes=cfg["model"]["num_classes"])
    conf_t = cfg["inference"]["conf_threshold"]
    iou_t = cfg["inference"]["iou_threshold"]

    model.eval()
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc=f"Evaluating [{args.split}]"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            decoded = model(images)
            B = images.shape[0]
            preds_list = []
            for b in range(B):
                dets = non_max_suppression(decoded[b:b+1], conf_t, iou_t)
                preds_list.append(dets[0])
            evaluator.update(preds_list, targets.cpu())

    results = evaluator.compute()
    metrics = results.to_dict()

    print("\n" + "═" * 50)
    print(f"  Evaluation Results [{args.split}]")
    print("═" * 50)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("═" * 50)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved: %s", args.output)


if __name__ == "__main__":
    main()
