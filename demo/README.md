# DroneVision MVP Demo Application

Welcome to the **DroneVision Demonstration Application**. This interface is built in pure Python using Gradio to showcase the capabilities of our custom-trained single-class drone detection system.

The application wraps the production-ready DroneVision inference pipeline, serving the pre-trained weights in a premium, dark-themed user interface optimized for recruiters, portfolio demonstrations, and deployment to Hugging Face Spaces.

---

## Overview

DroneVision solves the challenge of detecting very small drones (often ranging between 5 to 160 pixels) against complex backgrounds such as sky, clouds, urban environments, and birds. 

This demo provides a complete one-click workflow: upload an image, run inference using our custom PyTorch architecture, and receive clear bounding box overlays alongside an external detection count and confidence summary.

---

## Features

- **Single-Click Inference**: Uploading an image immediately triggers the custom YOLO-style detector.
- **Ultra-Clean Bounding Boxes**: Bounding boxes only feature sequential integer identifiers (`1`, `2`, `3`, ...) to prevent clutter on high-resolution images.
- **Comprehensive Detection Summary**: Precise metrics displayed outside the image frame:
  - **Drone Count**: Total number of detected drones.
  - **Inference Time**: Benchmark duration of the network's forward pass (in milliseconds).
  - **Execution Device**: Displays whether GPU acceleration (CUDA/MPS) or CPU fallback is active.
  - **Confidence Breakdown**: Individual confidence percentages (e.g. `94%`, `91%`) linked to each indexed box.
- **Example Selection**: Includes preloaded drone images for immediate trial testing.
- **Robust Error Handling**: Gracefully handles invalid formats, empty images, or hardware limitations without exposing technical tracebacks.

---

## Technology Stack

- **Frontend**: Gradio (Pure Python UI Engine)
- **Backend & Logic**: Python 3.11+
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV & Pillow
- **Deployment Target**: Hugging Face Spaces / Local Web Server

---

## Project Structure

All demo assets and logic are kept completely isolated in the `demo/` folder, ensuring the core research code remains untouched:

```text
demo/
├── app.py                      # Main Gradio application setup and event loop
├── predictor.py                # Wrapper isolating predictor instantiation & CPU fallback
├── ui.py                       # Pure-Python styling definitions and HTML builders
├── config.py                   # Configuration mappings and path variables
├── utils.py                    # Custom drawing methods and data formatting utilities
├── requirements.txt            # Package dependencies
├── README.md                   # Documentation (this file)
├── DEMO_VALIDATION_REPORT.md   # Deployment and correctness validation report
├── assets/                     # UI screenshots and visual assets
└── examples/                   # Preloaded sample images for immediate UI testing
```

---

## Installation

To run the demo application locally, clone the repository and set up a virtual environment:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 2. Install dependencies
pip install -r demo/requirements.txt
```

---

## How to Run

Launch the Gradio development server:

```bash
python demo/app.py
```

The terminal will print a local URL (typically `http://127.0.0.1:7860`). Open this address in your web browser.

---

## Example Workflow

1. Open the application in your browser.
2. Select one of the preloaded example images under **Example Images** or click the upload area to drop a custom drone photo.
3. The application runs the image through the custom detector.
4. Review the results:
   - The annotated image appears with green bounding boxes and numbered tags (e.g., `1`, `2`).
   - The **Summary Cards** display the total drone count, inference latency, and hardware device.
   - The **Confidence Table** details individual detection percentages.

---

## Visual Layout (Placeholder)

```text
+----------------------------------------------------------------+
|                         DRONEVISION                            |
|                 Custom Drone Detection Model                   |
|                 Built Completely From Scratch                  |
+------------------------------+---------------------------------+
|                              |                                 |
|      [ Upload Image ]        |     [ Annotated Output ]        |
|                              |                                 |
+------------------------------+---------------------------------+
|  DRONE COUNT   |   INFERENCE TIME   |    EXECUTION DEVICE      |
|       5        |       82 ms        |          CUDA            |
+----------------+--------------------+--------------------------+
|  ID            |  CONFIDENCE                                   |
|  1             |  94%                                          |
|  2             |  91%                                          |
+----------------------------------------------------------------+
```

---

## Known Limitations

- **Image Formats**: Supports `.jpg`, `.jpeg`, `.png`, and `.bmp`. Video file uploads or webcam streams are currently unsupported to maintain a lightweight footprint.
- **NMS Scaling**: Extremely cluttered frames with hundreds of small detections might experience brief CPU-bound NMS latency.

---

## Future Improvements

- **Batched Inference**: Support drag-and-drop batch processing.
- **Webcam Integration**: Real-time camera testing.
- **Optimized NMS**: Porting standard non-maximum suppression to GPU-based operators (`torchvision.ops.nms`).
