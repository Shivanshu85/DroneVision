from __future__ import annotations

import cv2
import numpy as np

def draw_custom_detections(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on the image with sequential integer labels only.
    
    Args:
        image: BGR image (HWC uint8 numpy array).
        boxes: (M, 4) normalized [x1, y1, x2, y2] coordinates.
        
    Returns:
        Annotated BGR image copy.
    """
    out = image.copy()
    h, w = out.shape[:2]
    
    for i, box in enumerate(boxes):
        # Scale normalized coordinates to pixel coordinates
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        
        # Clip coordinates to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Color: Bright Green in BGR
        color = (0, 255, 0)
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label (1-based index)
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Measure label size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Standard label placement: above the top-left of the bounding box
        label_x1 = x1
        label_y1 = y1 - text_h - 6
        label_x2 = x1 + text_w + 6
        label_y2 = y1
        
        text_x = x1 + 3
        text_y = y1 - 3
        
        # If the label goes off the top edge, position it inside the box instead
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = y1 + text_h + 6
            text_y = y1 + text_h + 3
            
        # Draw background filled rectangle (black for contrast)
        cv2.rectangle(out, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        # Draw fine green border around the label background
        cv2.rectangle(out, (label_x1, label_y1), (label_x2, label_y2), color, 1)
        # Draw the integer number
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color,
            font_thickness,
            lineType=cv2.LINE_AA
        )
        
    return out

def format_confidence_html(confidences: list[float]) -> str:
    """
    Format detection confidences as a clean, styled HTML table.
    
    Args:
        confidences: List of float confidence values.
        
    Returns:
        HTML string containing the formatted table.
    """
    if not confidences:
        return ""
        
    rows = []
    for i, conf in enumerate(confidences):
        pct = f"{int(round(conf * 100))}%"
        rows.append(f"""
        <tr>
            <td>{i + 1}</td>
            <td>{pct}</td>
        </tr>
        """)
        
    html = f"""
    <div class="table-container">
        <table class="confidence-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """
    return html
