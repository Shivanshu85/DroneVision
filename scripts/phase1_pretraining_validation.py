#!/usr/bin/env python3
"""
Phase 1 Pre-Training Validation Script for DroneVision.

Runs ALL validation checks required before full training:
    1. Environment Check (Python, CUDA, PyTorch, RAM, VRAM)
    2. Dataset Validation (image counts, label counts, format)
    3. Model Forward Pass Validation
    4. Loss Computation Validation
    5. Gradient Flow Validation
    6. Checkpoint Save/Load Validation

Generates: pretraining_validation_report.md

Usage:
    python scripts/phase1_pretraining_validation.py --config configs/phase1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# ── project path setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dronevision.data.collate import drone_collate_fn
from dronevision.data.dataset import DroneDataset
from dronevision.loss.detection_loss import DroneDetectionLoss
from dronevision.models.detector import DroneDetector
from dronevision.utils.config import load_config
from dronevision.utils.logger import configure_logging, get_logger
from dronevision.utils.reproducibility import get_device, set_seed

configure_logging()
logger = get_logger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


# ── helpers ──────────────────────────────────────────────────────────────────

def _label(ok: bool) -> str:
    return PASS if ok else FAIL


def _section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ── 1. Environment Check ─────────────────────────────────────────────────────

def check_environment() -> dict:
    _section("1. Environment Check")
    results: dict = {}

    # Python
    py_ver = sys.version.split()[0]
    py_ok = tuple(int(x) for x in py_ver.split(".")[:2]) >= (3, 10)
    print(f"  Python : {py_ver}  {_label(py_ok)}")
    results["python_version"] = py_ver
    results["python_ok"] = py_ok

    # PyTorch
    pt_ver = torch.__version__
    results["torch_version"] = pt_ver
    print(f"  PyTorch: {pt_ver}")

    # CUDA
    cuda_avail = torch.cuda.is_available()
    results["cuda_available"] = cuda_avail
    print(f"  CUDA   : {'available' if cuda_avail else 'NOT AVAILABLE'}  {_label(cuda_avail)}")

    if cuda_avail:
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            vram_ok = vram_gb >= 4.0
            print(f"  GPU {i}  : {props.name} | {vram_gb:.2f} GB VRAM  {_label(vram_ok)}")
            results[f"gpu_{i}_name"] = props.name
            results[f"gpu_{i}_vram_gb"] = round(vram_gb, 2)
            results[f"gpu_{i}_vram_ok"] = vram_ok
        results["cuda_version"] = torch.version.cuda
        print(f"  CUDA v : {torch.version.cuda}")
    else:
        results["cuda_version"] = None

    # RAM (Windows only)
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_gb = stat.ullTotalPhys / 1e9
        avail_ram_gb = stat.ullAvailPhys / 1e9
        ram_ok = total_ram_gb >= 8.0
        print(f"  RAM    : {total_ram_gb:.1f} GB total | {avail_ram_gb:.1f} GB free  {_label(ram_ok)}")
        results["ram_total_gb"] = round(total_ram_gb, 1)
        results["ram_ok"] = ram_ok
    except Exception as e:
        print(f"  RAM    : Could not detect ({e})")
        results["ram_total_gb"] = None
        results["ram_ok"] = None

    # Check packages
    pkgs = ["numpy", "cv2", "yaml", "mlflow", "tqdm", "scipy"]
    for pkg in pkgs:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"  {pkg:10s}: {ver}  ✅")
            results[f"pkg_{pkg}"] = ver
        except ImportError:
            print(f"  {pkg:10s}: MISSING  ❌")
            results[f"pkg_{pkg}"] = None

    results["environment_ok"] = cuda_avail and py_ok
    return results


# ── 2. Dataset Validation ─────────────────────────────────────────────────────

def check_dataset(cfg: dict) -> dict:
    _section("2. Dataset Validation")
    results: dict = {}

    data_cfg = cfg["data"]
    label_dir = Path(data_cfg.get("label_dir", "datasets/labels"))

    splits_ok = True
    for split in ["train", "val", "test"]:
        img_dir = Path(data_cfg[split])
        lbl_dir = label_dir / split

        if not img_dir.exists():
            print(f"  {split:5s}: img dir missing  ❌  {img_dir}")
            results[f"{split}_ok"] = False
            splits_ok = False
            continue
        if not lbl_dir.exists():
            print(f"  {split:5s}: lbl dir missing  ❌  {lbl_dir}")
            results[f"{split}_ok"] = False
            splits_ok = False
            continue

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        lbls = [p for p in lbl_dir.iterdir() if p.suffix == ".txt"]

        # Check alignment
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        missing_lbl = img_stems - lbl_stems
        extra_lbl = lbl_stems - img_stems

        # Count boxes and check for valid coords
        n_boxes = 0
        n_errors = 0
        n_empty = 0
        for lbl in lbls:
            lines = lbl.read_text().strip().splitlines()
            if not lines:
                n_empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    n_errors += 1
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if any(c < 0 or c > 1 for c in coords):
                    n_errors += 1
                elif cls_id != 0:
                    n_errors += 1  # wrong class
                else:
                    n_boxes += 1

        split_ok = (
            len(imgs) > 0
            and len(missing_lbl) == 0
            and n_errors < len(imgs) * 0.01  # <1% error rate
        )
        lbl_str = _label(split_ok)
        print(f"  {split:5s}: {len(imgs):5d} imgs | {len(lbls):5d} lbls | {n_boxes:6d} boxes"
              f" | {n_empty:4d} empty | {n_errors:3d} errors  {lbl_str}")
        if missing_lbl:
            print(f"         ⚠️  {len(missing_lbl)} images have no label file")

        results[f"{split}_images"] = len(imgs)
        results[f"{split}_labels"] = len(lbls)
        results[f"{split}_boxes"] = n_boxes
        results[f"{split}_errors"] = n_errors
        results[f"{split}_empty"] = n_empty
        results[f"{split}_ok"] = split_ok
        if not split_ok:
            splits_ok = False

    results["dataset_ok"] = splits_ok
    return results


# ── 3. Model Validation ───────────────────────────────────────────────────────

def check_model(cfg: dict, device: torch.device) -> dict:
    _section("3. Model Validation")
    results: dict = {}

    set_seed(42)
    model = DroneDetector(cfg)
    model.to(device)

    params = model.count_parameters()
    print(f"  Backbone params : {params['backbone']:,}")
    print(f"  Neck params     : {params['neck']:,}")
    print(f"  Head params     : {params['head']:,}")
    print(f"  Total params    : {params['total']:,}")
    results["total_params"] = params["total"]

    img_size = cfg["model"]["image_size"]
    batch = 2
    x = torch.randn(batch, 3, img_size, img_size, device=device)

    # Training forward pass
    model.train()
    try:
        raw = model(x)
        assert isinstance(raw, list) and len(raw) == 3, "Expected list of 3 scale tensors"
        strides = cfg["model"]["strides"]
        na = cfg["model"]["num_anchors"]
        nc = cfg["model"]["num_classes"]
        for i, (r, s) in enumerate(zip(raw, strides)):
            expected_h = img_size // s
            expected_shape = (batch, na, expected_h, expected_h, 5 + nc)
            assert r.shape == expected_shape, f"Scale {i}: got {r.shape}, expected {expected_shape}"
        print(f"  Train forward   : shapes {[list(r.shape) for r in raw]}  {PASS}")
        results["train_forward_ok"] = True
    except Exception as e:
        print(f"  Train forward   : {FAIL}  {e}")
        results["train_forward_ok"] = False
        results["model_ok"] = False
        return results

    # Eval forward pass
    model.eval()
    with torch.no_grad():
        try:
            decoded = model(x)
            nc5 = 5 + cfg["model"]["num_classes"]
            print(f"  Eval forward    : shape {list(decoded.shape)}  {PASS}")
            assert decoded.ndim == 3 and decoded.shape[0] == batch and decoded.shape[2] == nc5
            results["eval_forward_ok"] = True
        except Exception as e:
            print(f"  Eval forward    : {FAIL}  {e}")
            results["eval_forward_ok"] = False
            results["model_ok"] = False
            return results

    results["model_ok"] = results["train_forward_ok"] and results["eval_forward_ok"]
    return results


# ── 4. Loss Validation ────────────────────────────────────────────────────────

def check_loss(cfg: dict, device: torch.device) -> dict:
    _section("4. Loss Computation Validation")
    results: dict = {}

    set_seed(42)
    model = DroneDetector(cfg).to(device)
    criterion = DroneDetectionLoss(cfg)

    img_size = cfg["model"]["image_size"]
    batch = 2
    x = torch.randn(batch, 3, img_size, img_size, device=device)

    # Create synthetic targets: [batch_idx, cls, cx, cy, w, h]
    max_boxes = 10
    targets = torch.zeros(batch * max_boxes, 6, device=device)
    # Place a drone at center of each image
    for b in range(batch):
        targets[b * max_boxes, 0] = b       # batch index
        targets[b * max_boxes, 1] = 0       # class 0
        targets[b * max_boxes, 2] = 0.5     # cx
        targets[b * max_boxes, 3] = 0.5     # cy
        targets[b * max_boxes, 4] = 0.05    # w (small drone)
        targets[b * max_boxes, 5] = 0.05    # h

    model.train()
    try:
        raw = model(x)
        loss, metrics = criterion(raw, targets, model.anchors)
        is_nan = torch.isnan(loss)
        is_inf = torch.isinf(loss)
        loss_ok = not is_nan and not is_inf and loss.item() > 0
        print(f"  Loss value     : {loss.item():.4f}  {_label(loss_ok)}")
        print(f"  Box loss       : {metrics['box_loss']:.4f}")
        print(f"  Obj loss       : {metrics['obj_loss']:.4f}")
        print(f"  Cls loss       : {metrics['cls_loss']:.4f}")
        print(f"  NaN/Inf        : {'None  ✅' if not is_nan and not is_inf else 'DETECTED  ❌'}")
        results["loss_value"] = loss.item()
        results["loss_is_nan"] = bool(is_nan)
        results["loss_is_inf"] = bool(is_inf)
        results["loss_ok"] = loss_ok
    except Exception as e:
        print(f"  Loss compute   : {FAIL}  {e}")
        results["loss_ok"] = False
        return results

    return results


# ── 5. Gradient Flow Validation ───────────────────────────────────────────────

def check_gradients(cfg: dict, device: torch.device) -> dict:
    _section("5. Gradient Flow Validation")
    results: dict = {}

    set_seed(42)
    model = DroneDetector(cfg).to(device)
    criterion = DroneDetectionLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    img_size = cfg["model"]["image_size"]
    x = torch.randn(2, 3, img_size, img_size, device=device)
    targets = torch.zeros(2 * 10, 6, device=device)
    targets[0, 0] = 0; targets[0, 1] = 0; targets[0, 2:] = torch.tensor([0.5, 0.5, 0.05, 0.05])
    targets[10, 0] = 1; targets[10, 1] = 0; targets[10, 2:] = torch.tensor([0.3, 0.3, 0.05, 0.05])

    model.train()
    optimizer.zero_grad()
    raw = model(x)
    loss, _ = criterion(raw, targets, model.anchors)
    loss.backward()

    # Check gradients exist and are not all zero
    grad_ok = True
    zero_grad_layers = []
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_ok = False
            zero_grad_layers.append(f"{name}: NO GRAD")
        elif param.grad.abs().max() == 0:
            zero_grad_layers.append(f"{name}: zero grad")

    if zero_grad_layers:
        print(f"  Gradient check : {WARN}")
        for g in zero_grad_layers[:5]:
            print(f"    {g}")
    else:
        print(f"  Gradient check : All params have non-zero gradients  {PASS}")

    # Test optimizer step doesn't produce NaN weights
    optimizer.step()
    has_nan_weights = any(
        torch.isnan(p).any() for p in model.parameters()
    )
    print(f"  Post-step NaN  : {'None  ✅' if not has_nan_weights else 'NaN DETECTED  ❌'}")

    results["grad_flow_ok"] = grad_ok and not has_nan_weights
    results["post_step_nan"] = has_nan_weights
    return results


# ── 6. Checkpoint Save/Load Validation ────────────────────────────────────────

def check_checkpoint(cfg: dict, device: torch.device) -> dict:
    _section("6. Checkpoint Save/Load Validation")
    results: dict = {}

    ckpt_path = Path("runs/pretest/test_checkpoint.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    model = DroneDetector(cfg).to(device)

    # Save
    try:
        model.save_checkpoint(ckpt_path, epoch=0, metrics={"mAP50": 0.5})
        ckpt_size_kb = ckpt_path.stat().st_size / 1024
        print(f"  Checkpoint save : {ckpt_path} ({ckpt_size_kb:.1f} KB)  {PASS}")
        results["save_ok"] = True
        results["ckpt_size_kb"] = round(ckpt_size_kb, 1)
    except Exception as e:
        print(f"  Checkpoint save : {FAIL}  {e}")
        results["save_ok"] = False
        results["checkpoint_ok"] = False
        return results

    # Load
    try:
        model2, ckpt = DroneDetector.load_checkpoint(ckpt_path, cfg, device)
        print(f"  Checkpoint load : epoch={ckpt.get('epoch', '?')}  {PASS}")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"
        print(f"  Weight verify   : All weights match  {PASS}")
        results["load_ok"] = True
        results["weights_match"] = True
    except Exception as e:
        print(f"  Checkpoint load : {FAIL}  {e}")
        results["load_ok"] = False
        results["checkpoint_ok"] = False
        return results

    # Cleanup
    ckpt_path.unlink(missing_ok=True)
    results["checkpoint_ok"] = True
    return results


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(all_results: dict, cfg: dict) -> str:
    overall_ok = all(
        all_results.get(k, {}).get(f"{k.split('_')[0]}_ok", False)
        for k in all_results
    )
    lines = [
        "# DroneVision — Pre-Training Validation Report\n\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Config**: `configs/phase1.yaml`\n\n",
    ]

    env = all_results.get("environment", {})
    ds = all_results.get("dataset", {})
    model = all_results.get("model", {})
    loss = all_results.get("loss", {})
    grad = all_results.get("gradients", {})
    ckpt = all_results.get("checkpoint", {})

    lines.append("## Summary\n\n")
    lines.append("| Check | Status |\n|---|---|\n")
    lines.append(f"| Environment | {'✅ PASS' if env.get('environment_ok') else '❌ FAIL'} |\n")
    lines.append(f"| Dataset     | {'✅ PASS' if ds.get('dataset_ok') else '❌ FAIL'} |\n")
    lines.append(f"| Model       | {'✅ PASS' if model.get('model_ok') else '❌ FAIL'} |\n")
    lines.append(f"| Loss        | {'✅ PASS' if loss.get('loss_ok') else '❌ FAIL'} |\n")
    lines.append(f"| Gradients   | {'✅ PASS' if grad.get('grad_flow_ok') else '❌ FAIL'} |\n")
    lines.append(f"| Checkpoint  | {'✅ PASS' if ckpt.get('checkpoint_ok') else '❌ FAIL'} |\n\n")

    lines.append("## Environment Details\n\n")
    lines.append(f"- Python: `{env.get('python_version', '?')}`\n")
    lines.append(f"- PyTorch: `{env.get('torch_version', '?')}`\n")
    lines.append(f"- CUDA: `{env.get('cuda_version', 'N/A')}`\n")
    lines.append(f"- CUDA Available: `{env.get('cuda_available', False)}`\n")
    lines.append(f"- RAM: `{env.get('ram_total_gb', '?')} GB`\n")
    if env.get("gpu_0_name"):
        lines.append(f"- GPU: `{env['gpu_0_name']}` ({env.get('gpu_0_vram_gb', '?')} GB VRAM)\n")

    lines.append("\n## Dataset Details\n\n")
    lines.append("| Split | Images | Labels | Boxes | Empty | Errors |\n|---|---|---|---|---|---|\n")
    for split in ["train", "val", "test"]:
        lines.append(
            f"| {split} | {ds.get(f'{split}_images', '?')} "
            f"| {ds.get(f'{split}_labels', '?')} "
            f"| {ds.get(f'{split}_boxes', '?')} "
            f"| {ds.get(f'{split}_empty', '?')} "
            f"| {ds.get(f'{split}_errors', '?')} |\n"
        )

    lines.append("\n## Model Details\n\n")
    lines.append(f"- Total Parameters: `{model.get('total_params', '?'):,}`\n")
    lines.append(f"- Train Forward: `{'OK' if model.get('train_forward_ok') else 'FAILED'}`\n")
    lines.append(f"- Eval Forward: `{'OK' if model.get('eval_forward_ok') else 'FAILED'}`\n")

    lines.append("\n## Loss Details\n\n")
    lines.append(f"- Loss Value: `{loss.get('loss_value', '?'):.4f}`\n")
    lines.append(f"- NaN/Inf: `{'None' if not loss.get('loss_is_nan') and not loss.get('loss_is_inf') else 'DETECTED'}`\n")

    lines.append("\n## Gradient Flow\n\n")
    lines.append(f"- All gradients present: `{grad.get('grad_flow_ok', False)}`\n")
    lines.append(f"- Post-step NaN weights: `{grad.get('post_step_nan', '?')}`\n")

    lines.append("\n## Checkpoint\n\n")
    lines.append(f"- Save: `{'OK' if ckpt.get('save_ok') else 'FAILED'}`\n")
    lines.append(f"- Load: `{'OK' if ckpt.get('load_ok') else 'FAILED'}`\n")
    lines.append(f"- Size: `{ckpt.get('ckpt_size_kb', '?')} KB`\n")

    # Overall verdict
    all_checks = [
        env.get("environment_ok", False),
        ds.get("dataset_ok", False),
        model.get("model_ok", False),
        loss.get("loss_ok", False),
        grad.get("grad_flow_ok", False),
        ckpt.get("checkpoint_ok", False),
    ]
    all_pass = all(all_checks)
    lines.append("\n## Verdict\n\n")
    if all_pass:
        lines.append("**✅ ALL CHECKS PASSED — Cleared to proceed with training.**\n")
    else:
        lines.append("**❌ SOME CHECKS FAILED — Do NOT proceed with training until resolved.**\n")

    return "".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Pre-Training Validation")
    p.add_argument("--config", default="configs/phase1.yaml", type=Path)
    p.add_argument("--output", default="pretraining_validation_report.md", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(prefer_cuda=True)

    print(f"\n{'═' * 60}")
    print(f"  DroneVision — Phase 1 Pre-Training Validation")
    print(f"  Device: {device}")
    print(f"{'═' * 60}")

    all_results: dict = {}

    all_results["environment"] = check_environment()
    all_results["dataset"] = check_dataset(cfg)
    all_results["model"] = check_model(cfg, device)

    if all_results["model"].get("model_ok", False):
        all_results["loss"] = check_loss(cfg, device)
        all_results["gradients"] = check_gradients(cfg, device)
        all_results["checkpoint"] = check_checkpoint(cfg, device)
    else:
        print("\n⚠️  Skipping loss/gradient/checkpoint tests due to model failure.")
        all_results["loss"] = {"loss_ok": False}
        all_results["gradients"] = {"grad_flow_ok": False}
        all_results["checkpoint"] = {"checkpoint_ok": False}

    # Print summary
    _section("SUMMARY")
    checks = {
        "Environment" : all_results["environment"].get("environment_ok", False),
        "Dataset"     : all_results["dataset"].get("dataset_ok", False),
        "Model"       : all_results["model"].get("model_ok", False),
        "Loss"        : all_results["loss"].get("loss_ok", False),
        "Gradients"   : all_results["gradients"].get("grad_flow_ok", False),
        "Checkpoint"  : all_results["checkpoint"].get("checkpoint_ok", False),
    }
    all_pass = True
    for name, ok in checks.items():
        status = PASS if ok else FAIL
        print(f"  {name:15s}: {status}")
        if not ok:
            all_pass = False

    print(f"\n  {'CLEARED FOR TRAINING ✅' if all_pass else 'FIX FAILURES BEFORE TRAINING ❌'}")

    # Save report
    report = generate_report(all_results, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n  Report saved: {args.output.resolve()}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
