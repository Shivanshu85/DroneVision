"""Tests for model components: backbone, neck, head, detector."""

from __future__ import annotations

import pytest
import torch

from dronevision.models.backbone import DroneBackbone
from dronevision.models.neck import DroneNeck
from dronevision.models.head import DroneHead, DetectionScale
from dronevision.models.detector import DroneDetector
from dronevision.models.blocks import CBS, Bottleneck, SPPF


class TestBlocks:
    def test_cbs_forward(self):
        cbs = CBS(3, 32, k=3, s=1)
        x = torch.randn(2, 3, 64, 64)
        out = cbs(x)
        assert out.shape == (2, 32, 64, 64)

    def test_cbs_stride2(self):
        cbs = CBS(32, 64, k=3, s=2)
        x = torch.randn(2, 32, 64, 64)
        out = cbs(x)
        assert out.shape == (2, 64, 32, 32)

    def test_bottleneck_shortcut(self):
        bn = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(2, 64, 32, 32)
        out = bn(x)
        assert out.shape == (2, 64, 32, 32)

    def test_bottleneck_no_shortcut(self):
        bn = Bottleneck(64, 128, shortcut=False)
        x = torch.randn(2, 64, 32, 32)
        out = bn(x)
        assert out.shape == (2, 128, 32, 32)

    def test_sppf_output_shape(self):
        sppf = SPPF(256, 256, k=5)
        x = torch.randn(2, 256, 20, 20)
        out = sppf(x)
        assert out.shape == (2, 256, 20, 20)


class TestDroneBackbone:
    def test_output_scales(self, dev_cfg):
        bb = DroneBackbone(channels=[32, 64, 128, 256])
        x = torch.randn(2, 3, 416, 416)
        p3, p4, p5 = bb(x)
        assert p3.shape == (2, 128, 52, 52)  # 416/8
        assert p4.shape == (2, 256, 26, 26)  # 416/16
        assert p5.shape == (2, 256, 13, 13)  # 416/32

    def test_param_count_under_5m(self):
        bb = DroneBackbone(channels=[32, 64, 128, 256])
        params = sum(p.numel() for p in bb.parameters())
        assert params < 5_000_000, f"Backbone params {params/1e6:.2f}M exceeds 5M budget"

    def test_param_count_over_1m(self):
        bb = DroneBackbone(channels=[32, 64, 128, 256])
        params = sum(p.numel() for p in bb.parameters())
        assert params > 1_000_000, f"Backbone params {params/1e6:.2f}M is unexpectedly small"

    def test_gradients_flow(self):
        bb = DroneBackbone()
        x = torch.randn(1, 3, 416, 416, requires_grad=False)
        p3, p4, p5 = bb(x)
        p3.mean().backward()
        # Check that at least some parameters have gradients
        grad_norms = [p.grad.norm().item() for p in bb.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(np.isfinite(g) for g in grad_norms)

    def test_no_nan_in_output(self):
        import numpy as np
        bb = DroneBackbone()
        x = torch.randn(2, 3, 416, 416)
        p3, p4, p5 = bb(x)
        assert not torch.isnan(p3).any()
        assert not torch.isnan(p4).any()
        assert not torch.isnan(p5).any()


class TestDroneNeck:
    def test_output_shapes(self):
        neck = DroneNeck(in_channels=(128, 256, 256), out_channels=(64, 128, 128))
        p3 = torch.randn(2, 128, 52, 52)
        p4 = torch.randn(2, 256, 26, 26)
        p5 = torch.randn(2, 256, 13, 13)
        n3, n4, n5 = neck((p3, p4, p5))
        assert n3.shape == (2, 64, 52, 52)
        assert n4.shape == (2, 128, 26, 26)
        assert n5.shape == (2, 128, 13, 13)

    def test_no_nan_in_output(self):
        neck = DroneNeck()
        p3 = torch.randn(1, 128, 52, 52)
        p4 = torch.randn(1, 256, 26, 26)
        p5 = torch.randn(1, 256, 13, 13)
        n3, n4, n5 = neck((p3, p4, p5))
        assert not torch.isnan(n3).any()


class TestDroneHead:
    def test_output_shapes_3_scales(self):
        head = DroneHead(in_channels=(64, 128, 128), num_classes=1, num_anchors=3)
        n3 = torch.randn(2, 64, 52, 52)
        n4 = torch.randn(2, 128, 26, 26)
        n5 = torch.randn(2, 128, 13, 13)
        preds = head((n3, n4, n5))
        assert len(preds) == 3
        assert preds[0].shape == (2, 3, 52, 52, 6)  # 5 + nc=1
        assert preds[1].shape == (2, 3, 26, 26, 6)
        assert preds[2].shape == (2, 3, 13, 13, 6)

    def test_output_count_per_element(self):
        # Each anchor predicts [tx, ty, tw, th, obj, cls] = 6 values for nc=1
        head = DroneHead(in_channels=(64, 128, 128), num_classes=1, num_anchors=3)
        n3 = torch.randn(1, 64, 20, 20)
        n4 = torch.randn(1, 128, 10, 10)
        n5 = torch.randn(1, 128, 5, 5)
        preds = head((n3, n4, n5))
        assert preds[0].shape[-1] == 6  # 5 + 1 class


class TestDroneDetector:
    def test_training_mode_returns_list(self, dev_cfg):
        model = DroneDetector(dev_cfg)
        model.train()
        x = torch.randn(2, 3, 416, 416)
        preds = model(x)
        assert isinstance(preds, list)
        assert len(preds) == 3

    def test_eval_mode_returns_tensor(self, dev_cfg):
        model = DroneDetector(dev_cfg)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 416, 416)
            out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 3  # (B, N_total, 5+nc)

    def test_eval_output_in_valid_range(self, dev_cfg):
        model = DroneDetector(dev_cfg)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 416, 416)
            out = model(x)
        # obj and cls should be in (0, 1) after sigmoid
        assert out[0, :, 4].min() > 0.0
        assert out[0, :, 4].max() < 1.0

    def test_total_params_within_budget(self, dev_cfg):
        model = DroneDetector(dev_cfg)
        param_counts = model.count_parameters()
        total = param_counts["total"]
        assert total < 5_000_000, f"Total params {total/1e6:.2f}M exceeds 5M budget"
        assert total > 1_000_000, f"Total params {total/1e6:.2f}M unexpectedly small"

    def test_checkpoint_save_and_load(self, dev_cfg, tmp_path):
        model = DroneDetector(dev_cfg)
        ckpt_path = tmp_path / "test.pth"
        model.save_checkpoint(ckpt_path, epoch=1, metrics={"mAP50": 0.42})
        loaded_model, ckpt = DroneDetector.load_checkpoint(ckpt_path, dev_cfg, "cpu")
        assert ckpt["epoch"] == 1
        assert ckpt["metrics"]["mAP50"] == pytest.approx(0.42)

    def test_no_nan_in_eval_output(self, dev_cfg):
        model = DroneDetector(dev_cfg)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 416, 416)
            out = model(x)
        assert not torch.isnan(out).any()


# needed for gradient test above
import numpy as np
