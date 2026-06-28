#!/usr/bin/env python3
"""
Inference entry point for DroneVision.

Runs drone detection on one or more images and prints the drone count.
Optionally saves annotated output images.

Usage:
    # Single image
    python scripts/infer.py \
        --config configs/phase1.yaml \
        --weights runs/phase1/best.pth \
        --input path/to/image.jpg \
        --output-dir outputs/

    # Directory of images
    python scripts/infer.py \
        --config configs/phase1.yaml \
        --weights runs/phase1/best.pth \
        --input path/to/images/ \
        --output-dir outputs/ \
        --save-annotated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.inference.predictor import DronePredictor
from dronevision.inference.visualizer import save_annotated_image
from dronevision.utils.config import load_config
from dronevision.utils.logger import configure_logging, get_logger
from dronevision.utils.reproducibility import get_device

configure_logging()
logger = get_logger(__name__)

_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DroneVision inference")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path,
                   help="Image file or directory of images.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save annotated images.")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images with bounding boxes.")
    p.add_argument("--results-json", type=Path, default=None,
                   help="Save inference results to JSON.")
    return p.parse_args()


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(prefer_cuda=True)

    predictor = DronePredictor.from_checkpoint(args.weights, cfg, device)

    image_paths = collect_image_paths(args.input)
    logger.info("Running inference on %d images", len(image_paths))

    if args.output_dir and args.save_annotated:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    total_drones = 0

    print("\n" + "═" * 60)
    print(f"  DroneVision Inference | {len(image_paths)} images")
    print("═" * 60)

    for img_path in image_paths:
        result = predictor.predict_image(img_path)
        count = result["drone_count"]
        total_drones += count

        print(f"  {img_path.name:40s} | drones: {count:3d}")
        if count > 0 and len(result["confidences"]) > 0:
            max_conf = float(result["confidences"].max())
            print(f"    ↳ Max confidence: {max_conf:.3f}")

        # Save annotated image
        if args.save_annotated and args.output_dir:
            image = cv2.imread(str(img_path))
            if image is not None:
                out_path = args.output_dir / f"annotated_{img_path.name}"
                save_annotated_image(
                    image,
                    result["boxes"],
                    result["confidences"],
                    str(out_path),
                    drone_count=count,
                )

        all_results.append({
            "image": str(img_path),
            "drone_count": count,
            "boxes": result["boxes"].tolist(),
            "confidences": result["confidences"].tolist(),
        })

    print("═" * 60)
    print(f"  Total drones detected: {total_drones}")
    print(f"  Total images:          {len(image_paths)}")
    if len(image_paths) > 0:
        print(f"  Avg per image:         {total_drones / len(image_paths):.2f}")
    print("═" * 60)

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved: %s", args.results_json)


if __name__ == "__main__":
    main()
