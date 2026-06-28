# DroneVision

Custom single-class drone detection and counting system built from scratch with PyTorch.

> **Scope**: This project detects exactly **one class — Drone (Class 0)**. It is NOT a general object detector, classifier, or segmentation system.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .          # install dronevision as editable package
```

### 2. Convert dataset

```bash
python scripts/convert_voc_to_yolo.py \
    --source /path/to/DUT-Anti-UAV \
    --dest datasets/ \
    --splits train val test
```

### 3. Validate dataset

```bash
python scripts/validate_dataset.py --data datasets/
```

### 4. Analyze dataset statistics (optional but recommended)

```bash
python scripts/analyze_dataset.py --data datasets/ --suggest-anchors --img-size 640
```

### 5. Train

```bash
# Development run (fast, 10 epochs, 416×416)
python scripts/train.py --config configs/dev.yaml

# Phase 1 training (50 epochs, 640×640, mixed precision)
python scripts/train.py --config configs/phase1.yaml
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/phase1.yaml \
    --weights runs/phase1/best.pth \
    --split val
```

### 7. Run inference

```bash
# Single image
python scripts/infer.py \
    --config configs/phase1.yaml \
    --weights runs/phase1/best.pth \
    --input path/to/image.jpg \
    --save-annotated \
    --output-dir outputs/
```

---

## Architecture

```
Image (H×W×3)
    │
    ▼
DroneBackbone          ← ~2.3M params
 ├── Stage 0-1 (initial downsampling)
 ├── Stage 2 → P3 (H/8,  W/8,  128ch)   ← small drones
 ├── Stage 3 → P4 (H/16, W/16, 256ch)   ← medium drones
 └── Stage 4 + SPPF → P5 (H/32, W/32, 256ch) ← large drones
    │
    ▼
DroneNeck (FPN, top-down only)
 ├── N5 ← P5                              (128ch)
 ├── N4 ← concat[P4, up(N5)]             (128ch)
 └── N3 ← concat[P3, up(N4)]             (64ch)
    │
    ▼
DroneHead (3 scales × 3 anchors)
 ├── Scale 0 → (B, 3, H/8,  W/8,  6)    pred per anchor
 ├── Scale 1 → (B, 3, H/16, W/16, 6)
 └── Scale 2 → (B, 3, H/32, W/32, 6)
    │
    ▼
NMS → drone count
```

**Loss**: `λ_box × CIoU` + `λ_obj × BCE` + `λ_cls × BCE`

---

## Project Structure

```
DroneVision/
├── configs/
│   ├── dev.yaml          # 416×416, batch 8, 10 epochs (fast pipeline check)
│   └── phase1.yaml       # 640×640, batch 4, 50 epochs (primary training)
├── dronevision/
│   ├── data/
│   │   ├── converter.py  # Pascal VOC XML → YOLO
│   │   ├── dataset.py    # DroneDataset
│   │   ├── transforms.py # Letterbox, ToTensor
│   │   ├── augmentation.py # Mosaic, MixUp, ColorJitter, ...
│   │   └── collate.py    # drone_collate_fn
│   ├── models/
│   │   ├── blocks.py     # CBS, Bottleneck, SPPF
│   │   ├── backbone.py   # DroneBackbone (~2.3M)
│   │   ├── neck.py       # DroneNeck (FPN)
│   │   ├── head.py       # DroneHead
│   │   └── detector.py   # DroneDetector (full assembly)
│   ├── loss/
│   │   ├── iou_loss.py   # CIoU implementation
│   │   └── detection_loss.py  # Combined loss + target assignment
│   ├── engine/
│   │   ├── trainer.py    # DroneTrainer
│   │   ├── evaluator.py  # mAP50, mAP50-95
│   │   └── callbacks.py  # EarlyStopping, WarmupCosineScheduler
│   ├── inference/
│   │   ├── nms.py        # Non-maximum suppression
│   │   ├── predictor.py  # DronePredictor
│   │   └── visualizer.py # draw_detections
│   └── utils/
│       ├── logger.py
│       ├── config.py
│       ├── bbox.py
│       ├── anchors.py
│       └── reproducibility.py
├── scripts/
│   ├── convert_voc_to_yolo.py
│   ├── validate_dataset.py
│   ├── analyze_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_augmentation.py
│   ├── test_models.py
│   ├── test_loss.py
│   ├── test_nms.py
│   └── test_evaluator.py
├── datasets/             # populated by convert_voc_to_yolo.py
├── runs/                 # training checkpoints
├── mlruns/               # MLflow tracking data
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i5-14450HX or equivalent |
| RAM | 16 GB minimum |
| GPU | 6 GB VRAM (NVIDIA CUDA) |
| Storage | 5 GB for dataset + 1 GB for checkpoints |

Phase 1 is tuned for **batch size 4, mixed precision** to fit within 6 GB VRAM.
The model automatically selects CUDA → MPS → CPU.

---

## Performance Targets (Phase 1)

| Metric | Target |
|--------|--------|
| mAP50 | ≥ 0.80 |
| mAP50-95 | ≥ 0.55 |
| Inference speed | ≥ 15 FPS on GPU |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use synthetic data and run without the actual DUT Anti-UAV dataset installed.

---

## Experiment Tracking

Training metrics are logged to local MLflow:

```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

---

## Constraints

This project strictly prohibits:
- Ultralytics YOLO / YOLOv5 / YOLOv8 / YOLOv11
- Detectron2 / MMDetection / RT-DETR / GroundingDINO
- Any pretrained detection models
- Albumentations (all augmentation from scratch)
- Multi-class detection (only Drone = Class 0)
- Thermal fusion, radar fusion, multi-modal systems
- AWS deployment, TensorRT, tracking systems
