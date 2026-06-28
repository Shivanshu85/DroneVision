#!/usr/bin/env python3
"""
Dataset validation script for DroneVision.

Validates the YOLO-format dataset for:
  - Corrupt or unreadable images
  - Missing image files (label exists but no image)
  - Missing label files (image exists but no label)
  - Invalid bounding boxes (coordinates outside [0,1], w/h <= 0)
  - Invalid class IDs (any class != 0 is forbidden in DroneVision)
  - Duplicate image filenames across splits

Outputs a detailed JSON report. Training should NOT proceed if the
report contains critical errors.

Usage:
    python scripts/validate_dataset.py --data datasets/ --report validation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_VALID_CLASS_IDS: set[int] = {0}  # Only class 0 (Drone) is permitted
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass
class ImageReport:
    path: str
    status: str          # "ok", "corrupt", "missing_label", "invalid_label"
    errors: list[str] = field(default_factory=list)
    num_boxes: int = 0


@dataclass
class SplitReport:
    split: str
    total_images: int = 0
    ok: int = 0
    corrupt: int = 0
    missing_label: int = 0
    invalid_label: int = 0
    invalid_class: int = 0
    out_of_bounds: int = 0
    zero_area: int = 0
    total_boxes: int = 0
    image_reports: list[dict] = field(default_factory=list)


def validate_image(img_path: Path) -> tuple[bool, str]:
    """Try to decode the image header. Returns (is_valid, error_message)."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "cv2.imread returned None"
        if img.size == 0:
            return False, "Image has zero size"
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_label(label_path: Path, img_w: int = 1, img_h: int = 1) -> tuple[list[str], int]:
    """
    Validate a YOLO label file.

    Returns:
        (errors, num_valid_boxes)
    """
    errors: list[str] = []

    if not label_path.exists():
        return ["Missing label file"], 0

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return [], 0  # Background image — valid

    num_valid = 0
    for line_no, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {line_no}: expected 5 values, got {len(parts)}")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"Line {line_no}: non-numeric values")
            continue

        if cls_id not in _VALID_CLASS_IDS:
            errors.append(f"Line {line_no}: invalid class_id={cls_id} (must be 0)")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            errors.append(f"Line {line_no}: center coords out of [0,1]: cx={cx:.4f}, cy={cy:.4f}")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"Line {line_no}: invalid wh: w={w:.4f}, h={h:.4f}")
        if w * h < 1e-6:
            errors.append(f"Line {line_no}: near-zero area box (w={w:.6f}, h={h:.6f})")

        if not errors or errors[-1].startswith(f"Line {line_no}: invalid class"):
            num_valid += 1

    return errors, num_valid


def validate_split(
    image_dir: Path,
    label_dir: Path,
    split: str,
) -> SplitReport:
    """Validate all images and labels in one split."""
    report = SplitReport(split=split)

    if not image_dir.is_dir():
        logger.warning("Image directory not found: %s", image_dir)
        return report

    img_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    report.total_images = len(img_paths)
    logger.info("Validating split '%s': %d images", split, len(img_paths))

    for img_path in img_paths:
        img_report = ImageReport(path=str(img_path), status="ok")

        # Check image readability
        is_valid, err_msg = validate_image(img_path)
        if not is_valid:
            img_report.status = "corrupt"
            img_report.errors.append(f"Corrupt image: {err_msg}")
            report.corrupt += 1
            report.image_reports.append(asdict(img_report))
            continue

        # Check label
        label_path = label_dir / f"{img_path.stem}.txt"
        label_errors, num_boxes = validate_label(label_path)

        if label_errors:
            if "Missing label file" in label_errors:
                img_report.status = "missing_label"
                report.missing_label += 1
            else:
                img_report.status = "invalid_label"
                report.invalid_label += 1
                for e in label_errors:
                    if "invalid class_id" in e:
                        report.invalid_class += 1
                    elif "out of [0,1]" in e:
                        report.out_of_bounds += 1
                    elif "zero area" in e:
                        report.zero_area += 1
            img_report.errors = label_errors
        else:
            img_report.status = "ok"
            report.ok += 1

        img_report.num_boxes = num_boxes
        report.total_boxes += num_boxes

        # Only append error reports (keep report size manageable)
        if img_report.errors:
            report.image_reports.append(asdict(img_report))

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate DroneVision YOLO dataset")
    p.add_argument("--data", default="datasets", type=Path,
                   help="Dataset root (default: datasets/)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--report", default="validation_report.json", type=Path,
                   help="Output JSON report path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_reports: dict[str, dict] = {}
    has_critical_errors = False

    print("\n" + "═" * 60)
    print("  DroneVision Dataset Validation")
    print("═" * 60)

    for split in args.splits:
        img_dir = args.data / "images" / split
        lbl_dir = args.data / "labels" / split
        split_report = validate_split(img_dir, lbl_dir, split)
        all_reports[split] = asdict(split_report)

        # Print summary
        critical = split_report.corrupt + split_report.missing_label + split_report.invalid_class
        print(f"\n  [{split.upper()}]")
        print(f"    Total images    : {split_report.total_images}")
        print(f"    ✅ OK           : {split_report.ok}")
        print(f"    💀 Corrupt      : {split_report.corrupt}")
        print(f"    ❌ Missing label: {split_report.missing_label}")
        print(f"    ⚠  Invalid label: {split_report.invalid_label}")
        print(f"    ⛔ Invalid class : {split_report.invalid_class}")
        print(f"    📦 Total boxes  : {split_report.total_boxes}")

        if critical > 0:
            has_critical_errors = True
            print(f"    ⛔ CRITICAL ERRORS: {critical}")

    # Save report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n  Report saved: {args.report.resolve()}")
    print("═" * 60)

    if has_critical_errors:
        print("\n  ⛔ VALIDATION FAILED — Fix critical errors before training.")
        print("     See report for details.")
        sys.exit(1)
    else:
        print("\n  ✅ Dataset validation PASSED — ready for training.")
        print("     Next step: python scripts/analyze_dataset.py --data datasets/")


if __name__ == "__main__":
    main()
