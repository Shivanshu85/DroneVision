#!/usr/bin/env python3
"""
Convert DUT Anti-UAV Pascal VOC XML annotations to YOLO format.

This script MUST be run before any training or validation.

Usage:
    python scripts/convert_voc_to_yolo.py \
        --source /path/to/DUT-Anti-UAV \
        --dest datasets/ \
        --splits train val test

Output structure:
    datasets/
        images/
            train/  val/  test/
        labels/
            train/  val/  test/

Class mapping (fixed):
    UAV → 0  (the only permitted class in DroneVision)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.data.converter import VocToYoloConverter
from dronevision.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# DUT Anti-UAV uses "UAV" as the class name.
# Mapping to class 0 (Drone) is the ONLY permitted mapping.
CLASS_MAP: dict[str, int] = {"uav": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML → YOLO format for DroneVision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Root directory of the DUT Anti-UAV dataset.",
    )
    parser.add_argument(
        "--dest",
        default="datasets",
        type=Path,
        help="Output directory for YOLO-format dataset (default: datasets/).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to convert (default: train val test).",
    )
    parser.add_argument(
        "--annotation-dir",
        default="Annotations",
        help="Subdirectory name containing XML files (default: Annotations).",
    )
    parser.add_argument(
        "--image-dir",
        default="JPEGImages",
        help="Subdirectory name containing images (default: JPEGImages).",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Skip copying images (only generate label files).",
    )
    parser.add_argument(
        "--report",
        default="conversion_report.json",
        type=Path,
        help="Path to save the conversion report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        logger.error("Source directory not found: %s", args.source.resolve())
        sys.exit(1)

    converter = VocToYoloConverter(class_map=CLASS_MAP)
    all_reports: dict[str, dict] = {}

    for split in args.splits:
        logger.info("═" * 60)
        logger.info("Converting split: %s", split)
        logger.info("═" * 60)

        report = converter.convert(
            source_dir=args.source,
            dest_dir=args.dest,
            split=split,
            annotation_subdir=args.annotation_dir,
            image_subdir=args.image_dir,
            copy_images=not args.no_copy_images,
        )
        all_reports[split] = report.to_dict()

        # Print per-split summary
        print(f"\n{'='*50}")
        print(f"Split: {split}")
        print(f"  Total images:               {report.total_images}")
        print(f"  Successfully converted:     {report.converted}")
        print(f"  Skipped (no annotation):    {report.skipped_no_annotation}")
        print(f"  Skipped (no valid objects): {report.skipped_no_valid_objects}")
        print(f"  Skipped (missing image):    {report.skipped_missing_image}")
        print(f"  Errors:                     {len(report.errors)}")
        print(f"  Total boxes:                {report.total_boxes}")
        if report.unknown_classes:
            print(f"  ⚠ Unknown classes: {report.unknown_classes}")
        print(f"{'='*50}\n")

    # Save full report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    logger.info("Full conversion report saved: %s", args.report.resolve())

    # Final summary
    total_converted = sum(r["converted"] for r in all_reports.values())
    total_boxes = sum(r["total_boxes"] for r in all_reports.values())
    total_errors = sum(len(r["errors"]) for r in all_reports.values())
    print("\n🎯 Conversion complete!")
    print(f"   Total converted: {total_converted} images")
    print(f"   Total boxes:     {total_boxes}")
    print(f"   Errors:          {total_errors}")
    print(f"\n   Dataset ready at: {args.dest.resolve()}")
    print("   Next step: python scripts/validate_dataset.py --data datasets/")


if __name__ == "__main__":
    main()
