#!/usr/bin/env python3
"""
Dataset statistics and anchor analysis for DroneVision.

Generates:
  - Box scale distribution (small / medium / large by COCO thresholds)
  - Aspect ratio histogram
  - Per-split box count distribution
  - k-means anchor suggestions (9 clusters)
  - Summary JSON report

Usage:
    python scripts/analyze_dataset.py --data datasets/ --img-size 640
    python scripts/analyze_dataset.py --data datasets/ --suggest-anchors --img-size 640
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronevision.utils.anchors import suggest_anchors_kmeans
from dronevision.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# COCO thresholds (in pixel area at target image size)
_SMALL_MAX_AREA = 32 * 32      # < 32×32 pixels
_MEDIUM_MAX_AREA = 96 * 96     # 32–96 pixels


def collect_boxes(label_dir: Path, img_size: int) -> np.ndarray:
    """
    Read all YOLO label files and return box dimensions in pixel space.

    Args:
        label_dir: Path to labels/split directory.
        img_size:  Target image size for pixel conversion.

    Returns:
        (N, 2) array of [w_pixels, h_pixels] at img_size scale.
    """
    boxes: list[list[float]] = []
    txt_files = list(label_dir.glob("*.txt"))

    for txt in txt_files:
        try:
            data = np.loadtxt(str(txt), ndmin=2)
        except Exception:
            continue
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data[None]
        if data.shape[1] < 5:
            continue
        # wh in pixels at img_size
        w_px = data[:, 3] * img_size
        h_px = data[:, 4] * img_size
        for w, h in zip(w_px, h_px):
            boxes.append([w, h])

    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 2), dtype=np.float32)


def classify_scale(w: float, h: float) -> str:
    area = w * h
    if area < _SMALL_MAX_AREA:
        return "small"
    elif area < _MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def analyze_split(label_dir: Path, img_size: int) -> dict:
    boxes = collect_boxes(label_dir, img_size)
    n = len(boxes)

    if n == 0:
        return {
            "num_boxes": 0,
            "scale_distribution": {"small": 0, "medium": 0, "large": 0},
        }

    scale_counts = {"small": 0, "medium": 0, "large": 0}
    for w, h in boxes:
        scale_counts[classify_scale(w, h)] += 1

    areas = boxes[:, 0] * boxes[:, 1]
    aspects = boxes[:, 0] / (boxes[:, 1] + 1e-7)

    return {
        "num_boxes": n,
        "scale_distribution": scale_counts,
        "scale_pct": {
            k: f"{100*v/n:.1f}%"
            for k, v in scale_counts.items()
        },
        "box_w_mean_px": float(boxes[:, 0].mean()),
        "box_h_mean_px": float(boxes[:, 1].mean()),
        "box_w_std_px": float(boxes[:, 0].std()),
        "box_h_std_px": float(boxes[:, 1].std()),
        "area_mean_px2": float(areas.mean()),
        "area_median_px2": float(np.median(areas)),
        "area_p10_px2": float(np.percentile(areas, 10)),
        "area_p90_px2": float(np.percentile(areas, 90)),
        "aspect_mean": float(aspects.mean()),
        "aspect_median": float(np.median(aspects)),
    }


def suggest_anchors(all_boxes: np.ndarray, img_size: int) -> dict:
    """Run k-means and return formatted anchor suggestions."""
    if len(all_boxes) < 9:
        return {"error": f"Need at least 9 boxes for k-means, got {len(all_boxes)}"}

    try:
        clusters = suggest_anchors_kmeans(all_boxes, k=9, img_size=img_size)
    except Exception as e:
        return {"error": str(e)}

    # Sort by area and group into 3 scales (3 anchors each)
    anchors = clusters.tolist()
    return {
        "P3_stride8_small": [
            [round(a[0], 1), round(a[1], 1)] for a in anchors[0:3]
        ],
        "P4_stride16_medium": [
            [round(a[0], 1), round(a[1], 1)] for a in anchors[3:6]
        ],
        "P5_stride32_large": [
            [round(a[0], 1), round(a[1], 1)] for a in anchors[6:9]
        ],
        "yaml_format": {
            "anchors": [
                [[round(a[0]), round(a[1])] for a in anchors[0:3]],
                [[round(a[0]), round(a[1])] for a in anchors[3:6]],
                [[round(a[0]), round(a[1])] for a in anchors[6:9]],
            ]
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze DroneVision dataset statistics")
    p.add_argument("--data", default="datasets", type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--suggest-anchors", action="store_true",
                   help="Run k-means clustering on training boxes to suggest anchor sizes.")
    p.add_argument("--report", default="analysis_report.json", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  DroneVision Dataset Analysis")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print("═" * 60)

    analysis: dict = {"img_size": args.img_size, "splits": {}}

    all_train_boxes = np.zeros((0, 2), dtype=np.float32)

    for split in args.splits:
        lbl_dir = args.data / "labels" / split
        if not lbl_dir.is_dir():
            logger.warning("Label directory not found: %s", lbl_dir)
            continue

        stats = analyze_split(lbl_dir, args.img_size)
        analysis["splits"][split] = stats

        print(f"\n  [{split.upper()}] — {stats['num_boxes']} boxes")
        if stats["num_boxes"] > 0:
            dist = stats["scale_distribution"]
            pct = stats.get("scale_pct", {})
            print(f"    Scale distribution:")
            print(f"      Small  (<32×32px):  {dist['small']:5d}  ({pct.get('small','?')})")
            print(f"      Medium (32–96px):   {dist['medium']:5d}  ({pct.get('medium','?')})")
            print(f"      Large  (>96px):     {dist['large']:5d}  ({pct.get('large','?')})")
            print(f"    Box size (pixels):  W={stats['box_w_mean_px']:.1f}±{stats['box_w_std_px']:.1f}  "
                  f"H={stats['box_h_mean_px']:.1f}±{stats['box_h_std_px']:.1f}")
            print(f"    Area: median={stats['area_median_px2']:.0f}px²  "
                  f"P10={stats['area_p10_px2']:.0f}  P90={stats['area_p90_px2']:.0f}")
            print(f"    Aspect ratio: mean={stats['aspect_mean']:.2f}")

        # Collect training boxes for anchor suggestion
        if split == "train":
            all_train_boxes = collect_boxes(lbl_dir, args.img_size)

    # Anchor suggestion
    if args.suggest_anchors:
        print("\n  [ANCHOR SUGGESTIONS — k-means on training set]")
        anchor_result = suggest_anchors(all_train_boxes, args.img_size)
        analysis["anchor_suggestions"] = anchor_result

        if "error" in anchor_result:
            print(f"    ⚠ {anchor_result['error']}")
        else:
            print(f"    P3 (stride 8,  small):  {anchor_result['P3_stride8_small']}")
            print(f"    P4 (stride 16, medium): {anchor_result['P4_stride16_medium']}")
            print(f"    P5 (stride 32, large):  {anchor_result['P5_stride32_large']}")
            print("\n    Copy into configs/phase1.yaml:")
            yaml_anch = anchor_result["yaml_format"]["anchors"]
            print(f"    anchors:")
            for group in yaml_anch:
                print(f"      - {group}")

    # Global warning for small drone detection
    small_count = sum(
        s.get("scale_distribution", {}).get("small", 0)
        for s in analysis["splits"].values()
    )
    total_count = sum(s.get("num_boxes", 0) for s in analysis["splits"].values())
    if total_count > 0:
        small_pct = 100 * small_count / total_count
        print(f"\n  ⚡ Small drone ratio: {small_pct:.1f}% of all boxes")
        if small_pct > 50:
            print("     HIGH small-object ratio — P3 features are critical.")
            print("     Consider SAHI tiling if mAP50 < 0.50 after initial training.")

    # Save report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Report saved: {args.report.resolve()}")
    print("═" * 60)


if __name__ == "__main__":
    main()
