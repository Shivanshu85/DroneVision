"""
Visualization utilities for DroneVision inference results.

Draws bounding boxes and drone count overlays on images.
All coordinates are expected in normalized [0,1] format (xyxy).
"""

from __future__ import annotations

import cv2
import numpy as np

# Box color: bright green for high visibility against sky backgrounds
_BOX_COLOR: tuple[int, int, int] = (0, 255, 0)
_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
_COUNT_BG_COLOR: tuple[int, int, int] = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE_THICKNESS = 2
_FONT_SCALE = 0.6


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray | None = None,
    drone_count: int | None = None,
    color: tuple[int, int, int] = _BOX_COLOR,
) -> np.ndarray:
    """
    Draw bounding boxes and optional drone count on an image.

    Args:
        image:        HWC uint8 BGR image (original resolution).
        boxes:        (M, 4) float32 normalized [x1, y1, x2, y2] boxes.
        confidences:  (M,) float32 confidence scores (optional).
        drone_count:  Integer drone count to overlay (optional).
                      If None, computed as len(boxes).
        color:        BGR box color.

    Returns:
        Annotated copy of the input image (HWC uint8 BGR).
    """
    out = image.copy()
    h, w = out.shape[:2]
    count = drone_count if drone_count is not None else len(boxes)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        # Draw box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, _LINE_THICKNESS)

        # Draw confidence label
        if confidences is not None:
            label = f"Drone {confidences[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
            # Background rectangle for label
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 3),
                _FONT, _FONT_SCALE, _TEXT_COLOR, 1, cv2.LINE_AA,
            )

    # Draw drone count overlay in top-left corner
    count_text = f"Drones: {count}"
    (tw, th), _ = cv2.getTextSize(count_text, _FONT, 1.0, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), _COUNT_BG_COLOR, -1)
    cv2.putText(
        out, count_text, (10, th + 8),
        _FONT, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
    )

    return out


def save_annotated_image(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray | None,
    output_path: str,
    drone_count: int | None = None,
) -> None:
    """
    Draw detections on an image and save to disk.

    Args:
        image:       HWC uint8 BGR image.
        boxes:       (M, 4) normalized xyxy boxes.
        confidences: (M,) confidence scores.
        output_path: Destination file path.
        drone_count: Override for count display.
    """
    annotated = draw_detections(image, boxes, confidences, drone_count)
    cv2.imwrite(output_path, annotated)
