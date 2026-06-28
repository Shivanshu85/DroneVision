"""
Pascal VOC XML → YOLO format converter for DroneVision.

Converts DUT Anti-UAV dataset annotations from Pascal VOC XML format
to YOLO format (one .txt per image, one line per object).

YOLO label format:
    <class_id> <cx> <cy> <w> <h>
    All coordinates are normalized to [0.0, 1.0].
    class_id is always 0 (Drone) for this project.

Usage (via script):
    python scripts/convert_voc_to_yolo.py --source /path/to/DUT-Anti-UAV --dest datasets/

Usage (programmatic):
    from dronevision.data.converter import VocToYoloConverter
    converter = VocToYoloConverter(class_map={"UAV": 0})
    report = converter.convert(source_dir, dest_dir)
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dronevision.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image extensions
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass
class ConversionReport:
    """Summary of a dataset conversion run."""

    total_images: int = 0
    converted: int = 0
    skipped_no_annotation: int = 0
    skipped_no_valid_objects: int = 0
    skipped_missing_image: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    unknown_classes: set[str] = field(default_factory=set)
    total_boxes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "converted": self.converted,
            "skipped_no_annotation": self.skipped_no_annotation,
            "skipped_no_valid_objects": self.skipped_no_valid_objects,
            "skipped_missing_image": self.skipped_missing_image,
            "errors": self.errors,
            "unknown_classes": list(self.unknown_classes),
            "total_boxes": self.total_boxes,
        }


class VocToYoloConverter:
    """
    Converts Pascal VOC XML annotations to YOLO label format.

    Args:
        class_map: Mapping from VOC class name to integer class ID.
                   Example: {"UAV": 0}
    """

    def __init__(self, class_map: dict[str, int]) -> None:
        if not class_map:
            raise ValueError("class_map must not be empty")
        self.class_map = {k.strip().lower(): v for k, v in class_map.items()}
        logger.info("VocToYoloConverter initialized with class_map: %s", class_map)

    def convert(
        self,
        source_dir: str | Path,
        dest_dir: str | Path,
        split: str = "train",
        annotation_subdir: str = "Annotations",
        image_subdir: str = "JPEGImages",
        copy_images: bool = True,
    ) -> ConversionReport:
        """
        Convert all annotations in source_dir to YOLO format in dest_dir.

        Handles two common DUT Anti-UAV directory layouts:

        Layout A (flat):
            source_dir/
                JPEGImages/  (or images/)
                Annotations/ (or annotations/)

        Layout B (split-based):
            source_dir/
                train/
                    JPEGImages/
                    Annotations/
                val/
                    ...

        Args:
            source_dir:        Root of the VOC dataset.
            dest_dir:          Root of the output YOLO dataset.
            split:             "train", "val", or "test".
            annotation_subdir: Subdirectory name containing XML files.
            image_subdir:      Subdirectory name containing image files.
            copy_images:       Copy images to dest_dir if True.

        Returns:
            ConversionReport with conversion statistics.
        """
        source_dir = Path(source_dir)
        dest_dir = Path(dest_dir)
        report = ConversionReport()

        # Resolve annotation and image directories (Layout A or B)
        ann_dir, img_dir = self._resolve_dirs(
            source_dir, split, annotation_subdir, image_subdir
        )

        if ann_dir is None:
            logger.error(
                "Could not locate annotation directory in: %s", source_dir
            )
            report.errors.append(
                {"file": str(source_dir), "error": "Annotation directory not found"}
            )
            return report

        logger.info("Annotation dir: %s", ann_dir)
        logger.info("Image dir:      %s", img_dir)

        # Output directories
        out_label_dir = dest_dir / "labels" / split
        out_image_dir = dest_dir / "images" / split
        out_label_dir.mkdir(parents=True, exist_ok=True)
        if copy_images:
            out_image_dir.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(ann_dir.glob("*.xml"))
        logger.info("Found %d XML annotation files for split '%s'", len(xml_files), split)

        for xml_path in xml_files:
            report.total_images += 1
            try:
                self._convert_single(
                    xml_path=xml_path,
                    img_dir=img_dir,
                    out_label_dir=out_label_dir,
                    out_image_dir=out_image_dir,
                    copy_images=copy_images,
                    report=report,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing %s: %s", xml_path.name, exc)
                report.errors.append({"file": xml_path.name, "error": str(exc)})

        logger.info(
            "Conversion complete — converted: %d | skipped: %d | errors: %d",
            report.converted,
            report.skipped_no_annotation
            + report.skipped_no_valid_objects
            + report.skipped_missing_image,
            len(report.errors),
        )

        if report.unknown_classes:
            logger.warning(
                "Unknown class names encountered (not in class_map): %s",
                report.unknown_classes,
            )

        return report

    def _convert_single(
        self,
        xml_path: Path,
        img_dir: Path | None,
        out_label_dir: Path,
        out_image_dir: Path,
        copy_images: bool,
        report: ConversionReport,
    ) -> None:
        """Parse one XML file and write the corresponding YOLO .txt label."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image dimensions from XML
        size_node = root.find("size")
        if size_node is None:
            report.skipped_no_annotation += 1
            logger.debug("No <size> element in %s — skipping", xml_path.name)
            return

        img_w = int(size_node.findtext("width", default="0"))
        img_h = int(size_node.findtext("height", default="0"))

        if img_w <= 0 or img_h <= 0:
            report.skipped_no_annotation += 1
            logger.warning("Invalid image size in %s: %dx%d", xml_path.name, img_w, img_h)
            return

        # Locate source image
        filename_node = root.find("filename")
        stem = xml_path.stem

        img_path: Path | None = None
        if img_dir is not None:
            for ext in _IMAGE_EXTS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            # Also try exact filename from XML
            if img_path is None and filename_node is not None:
                candidate = img_dir / filename_node.text.strip()
                if candidate.exists():
                    img_path = candidate

        if copy_images and img_path is None:
            report.skipped_missing_image += 1
            logger.warning("Image not found for %s — skipping", xml_path.name)
            return

        # Parse object annotations
        yolo_lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()

            if name not in self.class_map:
                report.unknown_classes.add(name)
                continue

            class_id = self.class_map[name]
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))

            # Clamp to image bounds
            xmin = max(0.0, min(xmin, img_w))
            ymin = max(0.0, min(ymin, img_h))
            xmax = max(0.0, min(xmax, img_w))
            ymax = max(0.0, min(ymax, img_h))

            if xmax <= xmin or ymax <= ymin:
                logger.debug("Degenerate box in %s — skipping object", xml_path.name)
                continue

            # Convert to YOLO normalized format
            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            report.total_boxes += 1

        if not yolo_lines:
            # Background image — write an empty label file
            report.skipped_no_valid_objects += 1

        # Write label file (empty for background images)
        label_path = out_label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        # Optionally copy image
        if copy_images and img_path is not None:
            dest_img = out_image_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

        report.converted += 1

    @staticmethod
    def _resolve_dirs(
        source_dir: Path,
        split: str,
        annotation_subdir: str,
        image_subdir: str,
    ) -> tuple[Path | None, Path | None]:
        """
        Try to locate annotation and image directories under various layouts.

        Returns:
            (annotation_dir, image_dir) — either can be None if not found.
        """
        # Layout B: source/split/Annotations  (DUT common layout)
        b_ann = source_dir / split / annotation_subdir
        b_img = source_dir / split / image_subdir
        if b_ann.is_dir():
            return b_ann, b_img if b_img.is_dir() else None

        # Layout B alt: source/split/annotations  (lowercase)
        b_ann_low = source_dir / split / annotation_subdir.lower()
        b_img_low = source_dir / split / image_subdir.lower()
        if b_ann_low.is_dir():
            return b_ann_low, b_img_low if b_img_low.is_dir() else None

        # Layout A: source/Annotations  (flat)
        a_ann = source_dir / annotation_subdir
        a_img = source_dir / image_subdir
        if a_ann.is_dir():
            return a_ann, a_img if a_img.is_dir() else None

        # Layout A alt: lowercase
        a_ann_low = source_dir / annotation_subdir.lower()
        a_img_low = source_dir / image_subdir.lower()
        if a_ann_low.is_dir():
            return a_ann_low, a_img_low if a_img_low.is_dir() else None

        return None, None
