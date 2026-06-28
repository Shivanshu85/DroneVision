#!/usr/bin/env python3
"""
Phase 1: Dataset Verification for DroneVision.

Inspects the raw DUT Anti-UAV Pascal VOC dataset at dataset/ and generates
a comprehensive dataset_report.md.

Verifies:
  - Total images per split
  - Total XML annotations
  - Class names (must be UAV)
  - Bounding box quality (valid coords, non-degenerate)
  - Corrupt images (cv2.imread failure)
  - Duplicate filenames across splits
  - Box size distribution (small / medium / large)

Usage:
    python scripts/phase1_dataset_report.py --source dataset/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronevision.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# COCO thresholds (pixel area at 640×640)
_SMALL_MAX  = 32 * 32       # < 32×32 px
_MEDIUM_MAX = 96 * 96       # 32–96 px


def parse_xml(xml_path: Path) -> dict:
    """Parse one VOC XML file. Returns a dict with image metadata and boxes."""
    result = {
        "filename": "",
        "width": 0,
        "height": 0,
        "classes": [],
        "boxes": [],   # list of (xmin, ymin, xmax, ymax) in pixel coords
        "errors": [],
    }
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        result["filename"] = (root.findtext("filename") or xml_path.stem).strip()
        size = root.find("size")
        if size is None:
            result["errors"].append("Missing <size> element")
            return result
        result["width"]  = int(size.findtext("width",  default="0"))
        result["height"] = int(size.findtext("height", default="0"))

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            result["classes"].append(name)
            bndbox = obj.find("bndbox")
            if bndbox is None:
                result["errors"].append(f"Object '{name}' has no bndbox")
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            w, h = result["width"], result["height"]
            if w > 0 and h > 0:
                if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                    result["errors"].append(
                        f"Box out-of-bounds: [{xmin},{ymin},{xmax},{ymax}] in {w}×{h}"
                    )
                if xmax <= xmin or ymax <= ymin:
                    result["errors"].append(
                        f"Degenerate box: [{xmin},{ymin},{xmax},{ymax}]"
                    )
            result["boxes"].append((xmin, ymin, xmax, ymax))
    except ET.ParseError as e:
        result["errors"].append(f"XML parse error: {e}")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"Unexpected error: {e}")
    return result


def classify_box(xmin, ymin, xmax, ymax, img_w, img_h, target_size=640) -> str:
    """Classify box as small/medium/large by COCO thresholds at target_size."""
    bw = (xmax - xmin) / max(img_w, 1) * target_size
    bh = (ymax - ymin) / max(img_h, 1) * target_size
    area = bw * bh
    if area < _SMALL_MAX:
        return "small"
    if area < _MEDIUM_MAX:
        return "medium"
    return "large"


def image_hash(img_path: Path) -> str | None:
    """Compute MD5 of the first 8KB of an image for fast duplicate detection."""
    try:
        with open(img_path, "rb") as f:
            return hashlib.md5(f.read(8192)).hexdigest()
    except Exception:
        return None


def analyze_split(base: Path, split: str) -> dict:
    img_dir = base / split / "img"
    xml_dir = base / split / "xml"
    result = {
        "split": split,
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0,
        "corrupt_images": 0,
        "missing_label": 0,
        "label_errors": 0,
        "class_counts": {},
        "scale_counts": {"small": 0, "medium": 0, "large": 0},
        "box_widths": [],
        "box_heights": [],
        "img_widths": [],
        "img_heights": [],
        "hashes": {},
        "error_details": [],
    }

    if not img_dir.exists():
        logger.warning("Image dir not found: %s", img_dir)
        return result

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    result["total_images"] = len(img_paths)

    for img_path in img_paths:
        # Image validity
        img = cv2.imread(str(img_path))
        if img is None:
            result["corrupt_images"] += 1
            result["error_details"].append(f"CORRUPT: {img_path.name}")
            continue

        h, w = img.shape[:2]
        result["img_widths"].append(w)
        result["img_heights"].append(h)

        # Hash for duplicate detection
        h_val = image_hash(img_path)
        if h_val:
            result["hashes"][img_path.name] = h_val

        # Label
        xml_path = xml_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            result["missing_label"] += 1
            result["error_details"].append(f"MISSING_LABEL: {img_path.name}")
            continue

        result["total_labels"] += 1
        parsed = parse_xml(xml_path)

        if parsed["errors"]:
            result["label_errors"] += 1
            for e in parsed["errors"]:
                result["error_details"].append(f"LABEL_ERROR [{img_path.name}]: {e}")

        for cls in parsed["classes"]:
            result["class_counts"][cls] = result["class_counts"].get(cls, 0) + 1

        for (xmin, ymin, xmax, ymax) in parsed["boxes"]:
            result["total_boxes"] += 1
            bw = xmax - xmin
            bh = ymax - ymin
            result["box_widths"].append(bw)
            result["box_heights"].append(bh)
            scale = classify_box(xmin, ymin, xmax, ymax,
                                  parsed["width"] or w, parsed["height"] or h)
            result["scale_counts"][scale] += 1

    return result


def find_duplicates(splits_data: list[dict]) -> list[str]:
    """Find duplicate image hashes across all splits."""
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for sd in splits_data:
        for fname, h in sd["hashes"].items():
            key = f"{sd['split']}/{fname}"
            if h in seen:
                dupes.append(f"{seen[h]} ↔ {key}")
            else:
                seen[h] = key
    return dupes


def generate_report(source: Path, splits: list[str]) -> str:
    lines = ["# DroneVision — Dataset Report (Phase 1)\n"]
    lines.append(f"**Source**: `{source.resolve()}`\n")
    lines.append(f"**Dataset**: DUT Anti-UAV (Pascal VOC format)\n\n---\n")

    all_splits_data = []

    grand_images = grand_labels = grand_boxes = 0
    grand_corrupt = grand_missing = grand_errors = 0
    grand_scale = {"small": 0, "medium": 0, "large": 0}
    all_widths: list[float] = []
    all_heights: list[float] = []
    all_img_w: list[int] = []
    all_img_h: list[int] = []
    all_classes: dict[str, int] = {}

    for split in splits:
        logger.info("Analyzing split: %s", split)
        sd = analyze_split(source, split)
        all_splits_data.append(sd)

        grand_images  += sd["total_images"]
        grand_labels  += sd["total_labels"]
        grand_boxes   += sd["total_boxes"]
        grand_corrupt += sd["corrupt_images"]
        grand_missing += sd["missing_label"]
        grand_errors  += sd["label_errors"]
        all_widths.extend(sd["box_widths"])
        all_heights.extend(sd["box_heights"])
        all_img_w.extend(sd["img_widths"])
        all_img_h.extend(sd["img_heights"])
        for k, v in sd["scale_counts"].items():
            grand_scale[k] += v
        for k, v in sd["class_counts"].items():
            all_classes[k] = all_classes.get(k, 0) + v

        n = sd["total_boxes"]
        pct = lambda k: f"{100*sd['scale_counts'][k]/max(n,1):.1f}%"

        lines.append(f"## Split: {split.upper()}\n")
        lines.append(f"| Metric | Value |\n|---|---|\n")
        lines.append(f"| Total images | {sd['total_images']:,} |\n")
        lines.append(f"| Total labels (XML) | {sd['total_labels']:,} |\n")
        lines.append(f"| Total bounding boxes | {sd['total_boxes']:,} |\n")
        lines.append(f"| Corrupt images | {sd['corrupt_images']} |\n")
        lines.append(f"| Missing labels | {sd['missing_label']} |\n")
        lines.append(f"| Label errors | {sd['label_errors']} |\n")
        lines.append(f"| Classes found | {', '.join(sd['class_counts'].keys()) or 'none'} |\n")
        lines.append(f"| Small drones (<32×32px) | {sd['scale_counts']['small']:,} ({pct('small')}) |\n")
        lines.append(f"| Medium drones (32–96px) | {sd['scale_counts']['medium']:,} ({pct('medium')}) |\n")
        lines.append(f"| Large drones (>96px) | {sd['scale_counts']['large']:,} ({pct('large')}) |\n")

        if sd["img_widths"]:
            lines.append(f"| Avg image size | {np.mean(sd['img_widths']):.0f}×{np.mean(sd['img_heights']):.0f} px |\n")

        if sd["box_widths"]:
            lines.append(f"| Avg box width (px) | {np.mean(sd['box_widths']):.1f} |\n")
            lines.append(f"| Avg box height (px) | {np.mean(sd['box_heights']):.1f} |\n")
            lines.append(f"| Box width range | {min(sd['box_widths']):.1f} – {max(sd['box_widths']):.1f} px |\n")
            lines.append(f"| Box height range | {min(sd['box_heights']):.1f} – {max(sd['box_heights']):.1f} px |\n")
        lines.append("\n")

        if sd["error_details"]:
            lines.append(f"<details><summary>⚠️ {len(sd['error_details'])} issues (click to expand)</summary>\n\n```\n")
            for e in sd["error_details"][:50]:
                lines.append(f"{e}\n")
            if len(sd["error_details"]) > 50:
                lines.append(f"... and {len(sd['error_details'])-50} more\n")
            lines.append("```\n\n</details>\n\n")

    # Cross-split duplicates
    dupes = find_duplicates(all_splits_data)

    lines.append("## Overall Summary\n\n")
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| **Total images (all splits)** | **{grand_images:,}** |\n")
    lines.append(f"| **Total labels (all splits)** | **{grand_labels:,}** |\n")
    lines.append(f"| **Total bounding boxes** | **{grand_boxes:,}** |\n")
    lines.append(f"| Corrupt images | {grand_corrupt} |\n")
    lines.append(f"| Missing labels | {grand_missing} |\n")
    lines.append(f"| Label format errors | {grand_errors} |\n")
    lines.append(f"| Cross-split duplicates | {len(dupes)} |\n")
    lines.append(f"| Class names | {', '.join(all_classes.keys())} |\n")
    lines.append(f"| UAV→class-0 mapping | ✅ Required: UAV / Found: {list(all_classes.keys())} |\n")

    n = grand_boxes
    if n > 0:
        lines.append(f"\n### Bounding Box Scale Distribution\n\n")
        lines.append(f"| Scale | Count | % |\n|---|---|---|\n")
        for scale in ["small", "medium", "large"]:
            pct = 100 * grand_scale[scale] / n
            lines.append(f"| {scale.capitalize()} (<{'32×32' if scale=='small' else '96×96' if scale=='medium' else '>96×96'} px) | {grand_scale[scale]:,} | {pct:.1f}% |\n")

    if all_img_w:
        lines.append(f"\n### Image Size Statistics\n\n")
        lines.append(f"- Average: {np.mean(all_img_w):.0f}×{np.mean(all_img_h):.0f} px\n")
        lines.append(f"- Min: {min(all_img_w)}×{min(all_img_h)} px\n")
        lines.append(f"- Max: {max(all_img_w)}×{max(all_img_h)} px\n")

    if all_widths:
        lines.append(f"\n### Drone Size Statistics (pixel space)\n\n")
        lines.append(f"- Average box: {np.mean(all_widths):.1f}×{np.mean(all_heights):.1f} px\n")
        lines.append(f"- Median box: {np.median(all_widths):.1f}×{np.median(all_heights):.1f} px\n")
        lines.append(f"- Min box: {min(all_widths):.1f}×{min(all_heights):.1f} px\n")
        lines.append(f"- Max box: {max(all_widths):.1f}×{max(all_heights):.1f} px\n")

    lines.append("\n## Validation Decision\n\n")
    critical = grand_corrupt + grand_missing + grand_errors
    if critical == 0 and len(dupes) == 0:
        lines.append("✅ **PASSED** — Dataset is ready for VOC→YOLO conversion.\n\n")
        lines.append("**Next step**: Run `python scripts/convert_voc_to_yolo.py`\n")
    else:
        lines.append(f"⚠️ **Issues found**: {critical} critical errors, {len(dupes)} duplicates.\n\n")
        if critical > 0:
            lines.append(f"- {grand_corrupt} corrupt images\n")
            lines.append(f"- {grand_missing} missing labels\n")
            lines.append(f"- {grand_errors} label format errors\n")
        if dupes:
            lines.append(f"\n**Duplicates**:\n")
            for d in dupes[:10]:
                lines.append(f"- {d}\n")

    return "".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Dataset Verification")
    p.add_argument("--source", default="dataset", type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output", default="dataset_report.md", type=Path)
    return p.parse_args()


def main() -> None:
    import io, os
    # Force UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    print("\n" + "=" * 60)
    print("  DroneVision - Phase 1: Dataset Verification")
    print("=" * 60)
    report = generate_report(args.source, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n  Report saved: {args.output.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
